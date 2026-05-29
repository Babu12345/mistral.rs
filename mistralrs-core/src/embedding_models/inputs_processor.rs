#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, fmt::Debug, sync::Arc};

use anyhow::Result;
use candle_core::{Device, Tensor, WithDType};
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::{make_flash_params, FlashParams, PagedAttentionMeta},
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::Sequence,
    vision_models::{
        preprocessor_config::PreProcessorConfig, qwen3_vl::inputs_processor::Qwen3VLImageProcessor,
    },
};

fn _make_tensor_with_pad<D: WithDType>(
    x: Vec<Vec<D>>,
    max_len: usize,
    pad: D,
    device: &Device,
) -> Result<Tensor> {
    let mut padded_x = Vec::new();
    for mut x_i in x {
        assert!(x_i.len() <= max_len);
        x_i.extend([pad].repeat(max_len - x_i.len()));
        let shape = (x_i.len(),);
        padded_x.push(Tensor::from_vec(x_i, shape, device)?);
    }
    Tensor::cat(&padded_x[..], 0).map_err(anyhow::Error::msg)
}

pub struct InputMetadata {
    pub input: Tensor,
    pub flash_meta: FlashParams,
}

pub struct InnerInputProcessorOutput {
    pub inputs: InputMetadata,
    pub seq_indices: Vec<usize>,
}

// chunk_offset_toks is the number of tokens by which the tokens are offset,
// chunk_offset_toks / prompt_chunksize = number of batches
#[allow(clippy::too_many_arguments)]
pub fn make_prompt_chunk<T: WithDType + Debug>(
    chunk_offset_toks: usize,
    toks: Vec<&[T]>,
    device: &Device,
    mapper: Option<&dyn DeviceMapper>,
    has_causal_attention: bool,
    sliding_window: Option<usize>,
) -> Result<InputMetadata> {
    let max_len = toks
        .iter()
        .map(|seq| seq.len())
        .max()
        .expect("No sequences");
    let padding_tok = T::zero();
    // Pad each sequence by the padding token to the max len.
    let mut seqs_tensors = Vec::new();
    let flash_attn = crate::using_flash_attn();
    let mut seqlens_q = if flash_attn { vec![0] } else { Vec::new() };
    let mut seqlens_k = if flash_attn { vec![0] } else { Vec::new() };
    for ctxt in toks {
        let mut ctxt = ctxt.to_vec();
        ctxt.extend(std::iter::repeat_n(
            padding_tok,
            max_len.saturating_sub(ctxt.len()),
        ));

        if flash_attn {
            seqlens_q.push(ctxt.len() as u32);
            seqlens_k.push((ctxt.len() + chunk_offset_toks) as u32);
        }

        seqs_tensors.push(Tensor::new(ctxt, device).unwrap().unsqueeze(0).unwrap());
    }

    let flash_meta = if flash_attn {
        make_flash_params(
            device,
            mapper,
            &seqlens_q,
            &seqlens_k,
            sliding_window,
            has_causal_attention,
        )?
    } else {
        FlashParams::empty(has_causal_attention)
    };

    let input = Tensor::cat(&seqs_tensors, 0).unwrap();

    Ok(InputMetadata { input, flash_meta })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn get_prompt_input<T: WithDType + std::fmt::Debug>(
    toks: Vec<&[T]>,
    input_seqs: &[&mut Sequence],
    device: &Device,
    mapper: Option<&dyn DeviceMapper>,
    has_causal_attention: bool,
    sliding_window: Option<usize>,
) -> Result<InnerInputProcessorOutput> {
    let offset = input_seqs[0].token_offset();
    make_prompt_chunk(
        offset,
        toks,
        device,
        mapper,
        has_causal_attention,
        sliding_window,
    )
    .map(|inputs| InnerInputProcessorOutput {
        inputs,
        seq_indices: (0..input_seqs.len()).collect(),
    })
}

#[derive(Clone)]
pub struct VisionEmbeddingMeta {
    pub pixel_values: Tensor,
    pub image_grid_thw: Tensor,
    pub seqlens: Vec<usize>,
    pub continuous_img_pad: Vec<Vec<(usize, usize)>>,
    pub image_hashes: Vec<u64>,
}

#[derive(Clone)]
pub struct ModelInputs {
    pub input_ids: Tensor,
    pub flash_meta: FlashParams,
    pub vision: Option<VisionEmbeddingMeta>,
}

pub struct EmbeddingInputsProcessor {
    pub has_causal_attention: bool,
    pub vision_image_token_id: Option<u32>,
    pub vision_preprocessor_config: Option<Arc<PreProcessorConfig>>,
}

// Max edge (pixels) for input images before vision-tower preprocessing.
// Caps the patch count to keep the vision encoder's attention activation
// memory bounded on small GPUs - 512px square with patch=16, merge=2
// produces 256 patches and ~64 post-merge tokens, comfortable on 8GB Orin.
const MAX_IMAGE_EDGE: u32 = 512;

impl EmbeddingInputsProcessor {
    // Build VisionEmbeddingMeta from any images attached to the input sequences.
    // Returns None if vision isn't configured or no sequence has images.
    // Side effect: appends image-placeholder tokens to each sequence's token
    // stream so the model's forward has somewhere to splice the image
    // embeddings into the input_embeds tensor.
    fn build_vision_meta(
        &self,
        input_seqs: &mut [&mut Sequence],
        device: &Device,
    ) -> Result<Option<VisionEmbeddingMeta>> {
        let Some(image_token_id) = self.vision_image_token_id else {
            return Ok(None);
        };
        if !input_seqs.iter().any(|seq| seq.has_images()) {
            return Ok(None);
        }

        let preproc_cfg = self
            .vision_preprocessor_config
            .clone()
            .map(|a| (*a).clone())
            .unwrap_or_default();
        let qwen_proc = Qwen3VLImageProcessor { max_edge: None };
        let merge_size = preproc_cfg.merge_size.unwrap_or(2);

        let mut all_pixels = Vec::new();
        let mut all_grids: Vec<(u32, u32, u32)> = Vec::new();
        let mut all_hashes: Vec<u64> = Vec::new();
        let mut continuous_img_pad: Vec<Vec<(usize, usize)>> = Vec::with_capacity(input_seqs.len());
        let mut seqlens = Vec::with_capacity(input_seqs.len());

        for seq in input_seqs.iter_mut() {
            let images = seq.clone_images().unwrap_or_default();

            let mut seq_post_merge: usize = 0;
            let mut seq_grids: Vec<(u32, u32, u32)> = Vec::new();
            for img in &images {
                let resized = resize_max_edge(img, MAX_IMAGE_EDGE);
                let (w, h) = (resized.width(), resized.height());
                let (pixels, grid) = qwen_proc
                    .preprocess_inner(vec![resized], &preproc_cfg, device, (h, w))
                    .map_err(anyhow::Error::msg)?;
                let merge_sq = merge_size * merge_size;
                let patches = (grid.0 as usize) * (grid.1 as usize) * (grid.2 as usize);
                let post = patches / merge_sq.max(1);
                seq_post_merge += post;
                seq_grids.push(grid);
                all_pixels.push(pixels);
            }
            all_grids.extend(seq_grids);

            // Inject post-merge image tokens at the head of this sequence so the
            // splice logic has somewhere to land. continuous_img_pad span runs
            // [0, seq_post_merge); the original prompt tokens follow.
            let mut existing = seq.get_toks().to_vec();
            let mut new_toks: Vec<u32> = std::iter::repeat(image_token_id)
                .take(seq_post_merge)
                .collect();
            let span_end = new_toks.len();
            new_toks.append(&mut existing);
            seq.set_toks(new_toks.clone());

            seqlens.push(new_toks.len());
            continuous_img_pad.push(if seq_post_merge > 0 {
                vec![(0, span_end)]
            } else {
                Vec::new()
            });

            if let Some(hashes) = seq.image_hashes() {
                all_hashes.extend_from_slice(hashes);
            }
        }

        if all_pixels.is_empty() {
            return Ok(None);
        }
        let pixel_values = Tensor::cat(&all_pixels, 0).map_err(anyhow::Error::msg)?;
        let grid_flat: Vec<u32> = all_grids
            .iter()
            .flat_map(|&(t, h, w)| [t, h, w])
            .collect();
        let image_grid_thw =
            Tensor::from_vec(grid_flat, (all_grids.len(), 3), device).map_err(anyhow::Error::msg)?;

        Ok(Some(VisionEmbeddingMeta {
            pixel_values,
            image_grid_thw,
            seqlens,
            continuous_img_pad,
            image_hashes: all_hashes,
        }))
    }
}

fn resize_max_edge(img: &image::DynamicImage, max_edge: u32) -> image::DynamicImage {
    let (w, h) = (img.width(), img.height());
    let m = w.max(h);
    if m <= max_edge {
        return img.clone();
    }
    let scale = max_edge as f32 / m as f32;
    let new_w = ((w as f32 * scale) as u32).max(1);
    let new_h = ((h as f32 * scale) as u32).max(1);
    img.resize_exact(new_w, new_h, image::imageops::FilterType::CatmullRom)
}

impl InputsProcessor for EmbeddingInputsProcessor {
    fn process_inputs(
        &self,
        _: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        _is_xlora: bool,
        device: &Device,
        _no_kv_cache: bool,
        _last_n_context_len: Option<(usize, usize)>,
        _return_raw_logits: bool,
        _sliding_window: Option<usize>,
        _: Option<Arc<dyn Any>>,
        _paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput> {
        assert!(is_prompt);

        let metadata = get_prompt_input(
            input_seqs
                .iter()
                .map(|seq| seq.get_toks())
                .collect::<Vec<_>>(),
            input_seqs,
            device,
            mapper,
            self.has_causal_attention,
            None,
        )?;
        let InnerInputProcessorOutput {
            inputs:
                InputMetadata {
                    input: input_ids,
                    flash_meta,
                },
            seq_indices,
        } = metadata;
        let vision = self.build_vision_meta(input_seqs, device)?;
        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids,
            flash_meta,
            vision,
        });
        Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })
    }

    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Embedding
    }
}

pub struct EmbeddingProcessor {
    pub has_causal_attention: bool,
    pub vision_image_token_id: Option<u32>,
    pub vision_preprocessor_config: Option<Arc<PreProcessorConfig>>,
}

impl Processor for EmbeddingProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(EmbeddingInputsProcessor {
            has_causal_attention: self.has_causal_attention,
            vision_image_token_id: self.vision_image_token_id,
            vision_preprocessor_config: self.vision_preprocessor_config.clone(),
        })
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}
