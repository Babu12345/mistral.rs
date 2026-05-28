#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

// Qwen3-VL-Embedding-2B. Reuses the Qwen3-VL Instruct vision + text blocks
// (no LM head) and exposes them via the EmbeddingModel trait. Last-token
// pooling + L2 normalize happen in the surrounding EmbeddingPipeline.

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};
use std::sync::Arc;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::AttentionMask,
    device_map::DeviceMapper,
    layers::CausalMasker,
    layers_masker::{CausalMaskConfig, NotACache},
    paged_attention::AttentionImplementation,
    pipeline::{
        text_models_inputs_processor::FlashParams, EmbeddingModel, IsqModel, NormalLoadingMetadata,
    },
    vision_models::qwen3_vl::{
        config::Config, get_rope_index, text::Qwen3VLTextModel, vision::Qwen3VLVisionModel,
    },
};

pub struct Model {
    text: Qwen3VLTextModel,
    vision: Qwen3VLVisionModel,
    spatial_merge_size: usize,
    image_token_id: u32,
    video_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    device: Device,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let device = normal_loading_metadata.real_device.clone();
        let vision_vb = if vb.contains_tensor("vision_tower.patch_embed.proj.weight") {
            vb.pp("vision_tower")
        } else {
            vb.pp("model").pp("visual")
        };
        let vision =
            Qwen3VLVisionModel::new(&cfg.vision_config, vision_vb.set_device(device.clone()))?;
        let mut text_config = cfg.text_config.clone();
        if cfg.quantization_config.is_some() {
            text_config.quantization_config = cfg.quantization_config.clone();
        }
        let text = Qwen3VLTextModel::new(
            &text_config,
            vb.clone(),
            cfg.tie_word_embeddings,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            text,
            vision,
            spatial_merge_size: cfg.vision_config.spatial_merge_size,
            image_token_id: cfg.image_token_id,
            video_token_id: cfg.video_token_id,
            vision_start_token_id: cfg.vision_start_token_id,
            vision_end_token_id: cfg.vision_end_token_id,
            device,
        })
    }

    fn forward_text_only(
        &self,
        input_ids: &Tensor,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &NotACache,
            self.text.dtype,
            &CausalMaskConfig {
                sliding_window: self.text.cfg.sliding_window,
                ..Default::default()
            },
        )?;
        let input_embeds = self.text.embed_tokens(input_ids)?;
        let (bs, seq) = input_ids.dims2()?;
        // Plain 0..seq position ids broadcast across the 3 MRoPE channels;
        // text-only embedding has no image/video span to shift around.
        let position_ids = Tensor::arange(0i64, seq as i64, input_ids.device())?
            .reshape((1, 1, seq))?
            .broadcast_as((3, bs, seq))?
            .contiguous()?;
        self.text.forward_embeds_hidden_states(
            input_embeds,
            &attention_mask,
            &position_ids,
            None,
            flash_params,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_vision(
        &self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
        seqlens: Vec<usize>,
        continuous_img_pad: Vec<Vec<(usize, usize)>>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &NotACache,
            self.text.dtype,
            &CausalMaskConfig {
                sliding_window: self.text.cfg.sliding_window,
                ..Default::default()
            },
        )?;

        let mut input_embeds = self.text.embed_tokens(input_ids)?;
        let (batch_size, seq_len, hidden_dim) = input_embeds.dims3()?;
        let device = input_embeds.device().clone();

        let mut pixels = pixel_values.clone();
        let ndim = pixels.dims().len();
        if ndim > 2 {
            let last = pixels.dim(ndim - 1)?;
            pixels = pixels.reshape(((), last))?;
        }
        let (image_embeds, deepstack_image_embeds) =
            self.vision.forward(&pixels, image_grid_thw)?;
        let image_embeds = image_embeds.to_device(&device)?.to_dtype(self.text.dtype)?;
        let deepstack_image_embeds = deepstack_image_embeds
            .into_iter()
            .map(|t| t.to_device(&device)?.to_dtype(self.text.dtype))
            .collect::<Result<Vec<_>>>()?;

        let mut image_mask =
            Tensor::zeros((batch_size, seq_len), DType::F32, input_ids.device())?;
        let total_expected: usize = continuous_img_pad
            .iter()
            .flat_map(|spans| spans.iter().map(|(s, e)| e - s))
            .sum();
        if image_embeds.dim(0)? != total_expected {
            candle_core::bail!(
                "Image embedding length {} does not match placeholder tokens {}",
                image_embeds.dim(0)?,
                total_expected
            );
        }
        let mut offset = 0usize;
        for (batch, spans) in continuous_img_pad.iter().enumerate() {
            for &(start, end) in spans {
                let len = end - start;
                let chunk = image_embeds.narrow(0, offset, len)?;
                offset += len;
                input_embeds = input_embeds.slice_assign(
                    &[batch..batch + 1, start..end, 0..hidden_dim],
                    &chunk.unsqueeze(0)?,
                )?;
                let ones = Tensor::ones((1, len), DType::F32, input_ids.device())?;
                image_mask = image_mask.slice_assign(&[batch..batch + 1, start..end], &ones)?;
            }
        }
        let visual_pos_mask = image_mask.to_dtype(DType::U8)?;

        let max_seqlens = *seqlens.iter().max().unwrap_or(&seq_len);
        let mut ropeidx_attn_mask_bs = Vec::with_capacity(seqlens.len());
        for len in &seqlens {
            ropeidx_attn_mask_bs.push(Tensor::new(
                [vec![1f32; *len], vec![0f32; max_seqlens - len]].concat(),
                input_ids.device(),
            )?);
        }
        let ropeidx_attn_mask = Tensor::stack(&ropeidx_attn_mask_bs, 0)?;

        let (position_ids, _delta) = get_rope_index(
            input_ids,
            Some(image_grid_thw),
            None,
            &AttentionMask::Custom(ropeidx_attn_mask),
            self.spatial_merge_size,
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        )?;

        self.text.forward_embeds_hidden_states(
            input_embeds,
            &attention_mask,
            &position_ids,
            None,
            flash_params,
            Some(&visual_pos_mask),
            Some(&deepstack_image_embeds),
        )
    }
}

impl EmbeddingModel for Model {
    fn forward(&self, input_ids: &Tensor, flash_params: &FlashParams) -> Result<Tensor> {
        self.forward_text_only(input_ids, flash_params)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn forward_vision(
        &self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
        seqlens: Vec<usize>,
        continuous_img_pad: Vec<Vec<(usize, usize)>>,
        _image_hashes: &[u64],
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward_with_vision(
            input_ids,
            pixel_values,
            image_grid_thw,
            seqlens,
            continuous_img_pad,
            flash_params,
        )
    }
}

impl IsqModel for Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        // Delegate to the text submodel. Vision tower weights stay unquantized
        // (preserves retrieval quality, costs <200MB at bf16 for the 2B variant).
        self.text.get_layers()
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let mut tensors = self.text.residual_tensors();
        tensors.extend(self.vision.residual_tensors());
        tensors
    }

    fn imatrix_names(&self) -> Result<Vec<Option<String>>> {
        self.text.imatrix_names()
    }
}

impl AnyMoeBaseModelMixin for Model {}
