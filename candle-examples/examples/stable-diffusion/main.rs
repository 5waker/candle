#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use candle_transformers::models::stable_diffusion;

use anyhow::{Error as E, Result};
use candle::{DType, Device, IndexOp, Module, Tensor, D};
use clap::Parser;
use tokenizers::Tokenizer;

const GUIDANCE_SCALE: f64 = 7.5;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// 用于生成图像的提示词，又称为咒语、正咒。最好用","做分割，例如：a robot, like spide, red eyes。示例网站https://docs.qq.com/doc/DWFdSTHJtQWRzYk9k
    #[arg(
        long,
        default_value = "A very realistic photo of a rusty robot walking on a sandy beach"
    )]
    prompt: String,

    /// 反向提示词，又称为反咒。生成的图像中会尽量不包含该提示词。例如人类肖像的反咒：lowres, bad anatomy, bad hands, text, error, missing fingers
    #[arg(long, default_value = "")]
    uncond_prompt: String,

    /// 用CPU而不是GPU
    #[arg(long)]
    cpu: bool,

    /// 启用跟踪（生成trace-timestamp.json文件）
    #[arg(long)]
    tracing: bool,

    /// 生成图像的高度，单位像素
    #[arg(long)]
    height: Option<usize>,

    /// 生成图像的宽度，单位像素
    #[arg(long)]
    width: Option<usize>,

    /// UNet权重文件，格式为.safetensors。
    #[arg(long, value_name = "FILE")]
    unet_weights: Option<String>,

    /// CLIP权重文件，格式为.safetensors。
    #[arg(long, value_name = "FILE")]
    clip_weights: Option<String>,

    /// VAE权重文件，格式为.safetensors。
    #[arg(long, value_name = "FILE")]
    vae_weights: Option<String>,

    /// 器文件，用于分词。格式为.json。
    #[arg(long, value_name = "FILE")]
    tokenizer: Option<String>,

    /// 切片注意力的大小，或0表示自动切片（默认禁用）
    #[arg(long)]
    sliced_attention_size: Option<usize>,

    /// 运行SD的步数。很重要，某些模型22步更好，更少的步数也会更快完成。但是步数太少也会导致图像完全不可用。
    #[arg(long, default_value_t = 30)]
    n_steps: usize,

    /// 要生成的样本数。就是你要生成几张图。
    #[arg(long, default_value_t = 1)]
    num_samples: i64,

    /// 要生成的最终图像的名称。
    #[arg(long, value_name = "FILE", default_value = "sd_final.png")]
    final_image: String,

    /// 用于生成图像的模型版本。有vwaifu、v1-5、v2-1、xl四个版本可选。默认是v2.1。vwaifu是我增加的二次元版本（anything-5），其他是官方版本
    #[arg(long, value_enum, default_value = "v2-1")]
    sd_version: StableDiffusionVersion,

    /// 在每一步生成中间图像。没啥用且占IO，但是如果你想
    #[arg(long, action)]
    intermediary_images: bool,

    /// 使用flash_attn加速。默认是false。
    #[arg(long)]
    use_flash_attn: bool,

    /// 使用f16而不是f32。默认是false。半精度会生成的更快，但是质量可能会有影响，同时这也需要模型本身支持。anything-5不支持
    #[arg(long)]
    use_f16: bool,

    /// 图像生成图像，这里是图像路径
    #[arg(long, value_name = "FILE")]
    img2img: Option<String>,

    /// img2img强度，表示要转换初始图像的程度。该值必须介于0和1之间，值为1会丢弃初始图像信息。
    #[arg(long, default_value_t = 0.8)]
    img2img_strength: f64,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum StableDiffusionVersion {
    Vwaifu,
    V1_5,
    V2_1,
    Xl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFile {
    Tokenizer,
    Tokenizer2,
    Clip,
    Clip2,
    Unet,
    Vae,
}

impl StableDiffusionVersion {
    fn repo(&self) -> &'static str {
        match self {
            Self::Vwaifu => "stablediffusionapi/anything-v5",
            Self::Xl => "stabilityai/stable-diffusion-xl-base-1.0",
            Self::V2_1 => "stabilityai/stable-diffusion-2-1",
            Self::V1_5 => "runwayml/stable-diffusion-v1-5",
        }
    }

    fn unet_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::Vwaifu | Self::V1_5 | Self::V2_1 | Self::Xl => {
                if use_f16 {
                    "unet/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "unet/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    fn vae_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::Vwaifu | Self::V1_5 | Self::V2_1 | Self::Xl => {
                if use_f16 {
                    "vae/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "vae/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    fn clip_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::Vwaifu | Self::V1_5 | Self::V2_1 | Self::Xl => {
                if use_f16 {
                    "text_encoder/model.fp16.safetensors"
                } else {
                    "text_encoder/model.safetensors"
                }
            }
        }
    }

    fn clip2_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::Vwaifu | Self::V1_5 | Self::V2_1 | Self::Xl => {
                if use_f16 {
                    "text_encoder_2/model.fp16.safetensors"
                } else {
                    "text_encoder_2/model.safetensors"
                }
            }
        }
    }
}

impl ModelFile {
    fn get(
        &self,
        filename: Option<String>,
        version: StableDiffusionVersion,
        use_f16: bool,
    ) -> Result<std::path::PathBuf> {
        use hf_hub::api::sync::Api;
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let (repo, path) = match self {
                    Self::Tokenizer => {
                        let tokenizer_repo = match version {
                            StableDiffusionVersion::Vwaifu | StableDiffusionVersion::V1_5 | StableDiffusionVersion::V2_1 => {
                                "openai/clip-vit-base-patch32"
                            }
                            StableDiffusionVersion::Xl => {
                                // This seems similar to the patch32 version except some very small
                                // difference in the split regex.
                                "openai/clip-vit-large-patch14"
                            }
                        };
                        (tokenizer_repo, "tokenizer.json")
                    }
                    Self::Tokenizer2 => {
                        ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer.json")
                    }
                    Self::Clip => (version.repo(), version.clip_file(use_f16)),
                    Self::Clip2 => (version.repo(), version.clip2_file(use_f16)),
                    Self::Unet => (version.repo(), version.unet_file(use_f16)),
                    Self::Vae => {
                        // Override for SDXL when using f16 weights.
                        // See https://github.com/huggingface/candle/issues/1060
                        if version == StableDiffusionVersion::Xl && use_f16 {
                            (
                                "madebyollin/sdxl-vae-fp16-fix",
                                "diffusion_pytorch_model.safetensors",
                            )
                        } else {
                            (version.repo(), version.vae_file(use_f16))
                        }
                    }
                };
                let filename = Api::new()?.model(repo.to_string()).get(path)?;
                Ok(filename)
            }
        }
    }
}

fn output_filename(
    basename: &str,
    sample_idx: i64,
    num_samples: i64,
    timestep_idx: Option<usize>,
) -> String {
    let filename = if num_samples > 1 {
        match basename.rsplit_once('.') {
            None => format!("{basename}.{sample_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}.{sample_idx}.{extension}")
            }
        }
    } else {
        basename.to_string()
    };
    match timestep_idx {
        None => filename,
        Some(timestep_idx) => match filename.rsplit_once('.') {
            None => format!("{filename}-{timestep_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}-{timestep_idx}.{extension}")
            }
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn text_embeddings(
    prompt: &str,
    uncond_prompt: &str,
    tokenizer: Option<String>,
    clip_weights: Option<String>,
    sd_version: StableDiffusionVersion,
    sd_config: &stable_diffusion::StableDiffusionConfig,
    use_f16: bool,
    device: &Device,
    dtype: DType,
    first: bool,
) -> Result<Tensor> {
    let tokenizer_file = if first {
        ModelFile::Tokenizer
    } else {
        ModelFile::Tokenizer2
    };
    let tokenizer = tokenizer_file.get(tokenizer, sd_version, use_f16)?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let pad_id = match &sd_config.clip.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    println!("Running with prompt \"{prompt}\".");
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    while tokens.len() < sd_config.clip.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

    let mut uncond_tokens = tokenizer
        .encode(uncond_prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    while uncond_tokens.len() < sd_config.clip.max_position_embeddings {
        uncond_tokens.push(pad_id)
    }
    let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), device)?.unsqueeze(0)?;

    println!("Building the Clip transformer.");
    let clip_weights_file = if first {
        ModelFile::Clip
    } else {
        ModelFile::Clip2
    };
    let clip_weights = clip_weights_file.get(clip_weights, sd_version, false)?;
    let clip_config = if first {
        &sd_config.clip
    } else {
        sd_config.clip2.as_ref().unwrap()
    };
    let text_model =
        stable_diffusion::build_clip_transformer(clip_config, clip_weights, device, DType::F32)?;
    let text_embeddings = text_model.forward(&tokens)?;
    let uncond_embeddings = text_model.forward(&uncond_tokens)?;
    let text_embeddings = Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(dtype)?;
    Ok(text_embeddings)
}

fn image_preprocess<T: AsRef<std::path::Path>>(path: T) -> anyhow::Result<Tensor> {
    let img = image::io::Reader::open(path)?.decode()?;
    let (height, width) = (img.height() as usize, img.width() as usize);
    let height = height - height % 32;
    let width = width - width % 32;
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?
        .unsqueeze(0)?;
    Ok(img)
}

fn run(args: Args) -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let Args {
        prompt,
        uncond_prompt,
        cpu,
        height,
        width,
        n_steps,
        tokenizer,
        final_image,
        sliced_attention_size,
        num_samples,
        sd_version,
        clip_weights,
        vae_weights,
        unet_weights,
        tracing,
        use_f16,
        use_flash_attn,
        img2img,
        img2img_strength,
        ..
    } = args;

    if !(0. ..=1.).contains(&img2img_strength) {
        anyhow::bail!("img2img-strength should be between 0 and 1, got {img2img_strength}")
    }

    let _guard = if tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let dtype = if use_f16 { DType::F16 } else { DType::F32 };
    let sd_config = match sd_version {
        StableDiffusionVersion::Vwaifu => {
            stable_diffusion::StableDiffusionConfig::v1_5(sliced_attention_size, height, width)
        }
        StableDiffusionVersion::V1_5 => {
            stable_diffusion::StableDiffusionConfig::v1_5(sliced_attention_size, height, width)
        }
        StableDiffusionVersion::V2_1 => {
            stable_diffusion::StableDiffusionConfig::v2_1(sliced_attention_size, height, width)
        }
        StableDiffusionVersion::Xl => {
            stable_diffusion::StableDiffusionConfig::sdxl(sliced_attention_size, height, width)
        }
    };

    let scheduler = sd_config.build_scheduler(n_steps)?;
    let device = candle_examples::device(cpu)?;

    let which = match sd_version {
        StableDiffusionVersion::Xl => vec![true, false],
        _ => vec![true],
    };
    let text_embeddings = which
        .iter()
        .map(|first| {
            text_embeddings(
                &prompt,
                &uncond_prompt,
                tokenizer.clone(),
                clip_weights.clone(),
                sd_version,
                &sd_config,
                use_f16,
                &device,
                dtype,
                *first,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let text_embeddings = Tensor::cat(&text_embeddings, D::Minus1)?;
    println!("{text_embeddings:?}");

    println!("Building the autoencoder.");
    let vae_weights = ModelFile::Vae.get(vae_weights, sd_version, use_f16)?;
    let vae = sd_config.build_vae(&vae_weights, &device, dtype)?;
    let init_latent_dist = match &img2img {
        None => None,
        Some(image) => {
            let image = image_preprocess(image)?.to_device(&device)?;
            Some(vae.encode(&image)?)
        }
    };
    println!("Building the unet.");
    let unet_weights = ModelFile::Unet.get(unet_weights, sd_version, use_f16)?;
    let unet = sd_config.build_unet(&unet_weights, &device, 4, use_flash_attn, dtype)?;

    let t_start = if img2img.is_some() {
        n_steps - (n_steps as f64 * img2img_strength) as usize
    } else {
        0
    };
    let bsize = 1;
    for idx in 0..num_samples {
        let timesteps = scheduler.timesteps();
        let latents = match &init_latent_dist {
            Some(init_latent_dist) => {
                let latents = (init_latent_dist.sample()? * 0.18215)?.to_device(&device)?;
                if t_start < timesteps.len() {
                    let noise = latents.randn_like(0f64, 1f64)?;
                    scheduler.add_noise(&latents, noise, timesteps[t_start])?
                } else {
                    latents
                }
            }
            None => {
                let latents = Tensor::randn(
                    0f32,
                    1f32,
                    (bsize, 4, sd_config.height / 8, sd_config.width / 8),
                    &device,
                )?;
                // scale the initial noise by the standard deviation required by the scheduler
                (latents * scheduler.init_noise_sigma())?
            }
        };
        let mut latents = latents.to_dtype(dtype)?;

        println!("starting sampling");
        for (timestep_index, &timestep) in timesteps.iter().enumerate() {
            if timestep_index < t_start {
                continue;
            }
            let start_time = std::time::Instant::now();
            let latent_model_input = Tensor::cat(&[&latents, &latents], 0)?;

            let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)?;
            let noise_pred =
                unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;
            let noise_pred = noise_pred.chunk(2, 0)?;
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
            let noise_pred =
                (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * GUIDANCE_SCALE)?)?;
            latents = scheduler.step(&noise_pred, timestep, &latents)?;
            let dt = start_time.elapsed().as_secs_f32();
            println!("step {}/{n_steps} done, {:.2}s", timestep_index + 1, dt);

            if args.intermediary_images {
                let image = vae.decode(&(&latents / 0.18215)?)?;
                let image = ((image / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
                let image = (image * 255.)?.to_dtype(DType::U8)?.i(0)?;
                let image_filename =
                    output_filename(&final_image, idx + 1, num_samples, Some(timestep_index + 1));
                candle_examples::save_image(&image, image_filename)?
            }
        }

        println!(
            "Generating the final image for sample {}/{}.",
            idx + 1,
            num_samples
        );
        let image = vae.decode(&(&latents / 0.18215)?)?;
        let image = ((image / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
        let image = (image.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?.i(0)?;
        let image_filename = output_filename(&final_image, idx + 1, num_samples, None);
        candle_examples::save_image(&image, image_filename)?
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}
