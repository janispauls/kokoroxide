use super::voice::VoiceStyle;
use crate::espeak::EspeakIpaTokenizer;
use ort::ep::ExecutionProviderDispatch;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use ort::{self, inputs};
use std::error::Error;
use std::io::Cursor;
use std::path::Path;

pub struct TTSConfig {
    pub model_path: String,
    pub tokenizer_path: String,
    pub max_length: usize,
    pub sample_rate: u32,
    pub graph_level: GraphOptimizationLevel,
    pub execution_provider: Vec<ExecutionProviderDispatch>,
}

impl TTSConfig {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Self {
        TTSConfig {
            model_path: model_path.to_string(),
            tokenizer_path: tokenizer_path.to_string(),
            max_length: 512,
            sample_rate: 24000,
            graph_level: GraphOptimizationLevel::Level3,
            execution_provider: vec![],
        }
    }

    pub fn with_max_tokens_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    pub fn with_graph_optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
        self.graph_level = level;
        self
    }

    pub fn with_execution_providers(mut self, providers: Vec<ExecutionProviderDispatch>) -> Self {
        self.execution_provider = providers;
        self
    }
}

pub struct GeneratedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub duration_seconds: f32,
}

impl GeneratedAudio {
    pub fn save_to_wav<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        let bytes = self.to_wav_bytes()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    pub fn to_wav_bytes(&self) -> Result<Vec<u8>, Box<dyn Error>> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec)?;

            // Add 0.1 seconds of silence at the beginning
            let silence_samples = (self.sample_rate as f32 * 0.1) as usize;
            for _ in 0..silence_samples {
                writer.write_sample(0i16)?;
            }

            // Write the actual audio
            for &sample in &self.samples {
                // Clamp to prevent overflow
                let clamped = sample.clamp(-1.0, 1.0);
                let amplitude = (clamped * i16::MAX as f32) as i16;
                writer.write_sample(amplitude)?;
            }

            // Add 0.1 seconds of silence at the end
            for _ in 0..silence_samples {
                writer.write_sample(0i16)?;
            }

            writer.finalize()?;
        }

        Ok(cursor.into_inner())
    }
}

pub struct KokoroTTS {
    session: Session,
    tokenizer: EspeakIpaTokenizer,
    sample_rate: u32,
}

impl KokoroTTS {
    pub fn with_config(config: TTSConfig) -> Result<Self, Box<dyn Error>> {
        let TTSConfig {
            model_path,
            tokenizer_path,
            max_length,
            sample_rate,
            graph_level,
            execution_provider,
        } = config;

        //let env = Arc::new(ort::init().with_name("kokoro_tts"));
        //let env = Arc::new(Environment::builder().with_name("kokoro_tts").build()?);

        let optimization = match graph_level {
            GraphOptimizationLevel::Disable => GraphOptimizationLevel::Disable,
            GraphOptimizationLevel::Level1 => GraphOptimizationLevel::Level1,
            GraphOptimizationLevel::Level2 => GraphOptimizationLevel::Level2,
            GraphOptimizationLevel::Level3 => GraphOptimizationLevel::Level3,
            GraphOptimizationLevel::All => GraphOptimizationLevel::All,
        };

        let mut builder = Session::builder()?
            .with_optimization_level(optimization)?
            .with_parallel_execution(true)?;

        if !execution_provider.is_empty() {
            builder = builder.with_execution_providers(&execution_provider)?;
        }

        let session = builder.commit_from_file(&model_path)?;

        let tokenizer_content = std::fs::read_to_string(&tokenizer_path)?;
        let tokenizer_json: serde_json::Value = serde_json::from_str(&tokenizer_content)?;
        let vocab_obj = tokenizer_json["model"]["vocab"]
            .as_object()
            .ok_or("No vocab found in tokenizer.json")?;

        let mut vocab = std::collections::HashMap::new();
        for (token, id) in vocab_obj {
            vocab.insert(token.clone(), id.as_i64().unwrap_or(0));
        }

        let tokenizer = EspeakIpaTokenizer::new(vocab)?.with_model_max_length(max_length);

        Ok(KokoroTTS {
            session,
            tokenizer,
            sample_rate,
        })
    }

    pub fn generate_speech_from_phonemes(
        &mut self,
        phonemes: &str,
        voice_style: &VoiceStyle,
        speed: f32,
    ) -> Result<GeneratedAudio, Box<dyn Error>> {
        let tokens = self.tokenizer.encode_phonemes(phonemes, None)?;

        self.generate_from_tokens(&tokens, voice_style, speed)
    }

    pub fn generate_speech(
        &mut self,
        text: &str,
        voice_style: &VoiceStyle,
        speed: f32,
    ) -> Result<GeneratedAudio, Box<dyn Error>> {
        let tokens = self.tokenizer.encode(text, None)?;

        self.generate_from_tokens(&tokens, voice_style, speed)
    }

    pub fn generate_from_tokens(
        &mut self,
        tokens: &[i64],
        voice_style: &VoiceStyle,
        speed: f32,
    ) -> Result<GeneratedAudio, Box<dyn Error>> {
        let style_vector = voice_style.get_style_vector_for_token_length(tokens.len(), 256);

        let input_ids_tensor = Tensor::from_array(([1, tokens.len()], tokens.to_vec()))?;
        let style_tensor = Tensor::from_array(([1usize, 256], style_vector))?;
        let speed_tensor = Tensor::from_array(([1usize], vec![speed]))?;

        let outputs = self
            .session
            .run(inputs![input_ids_tensor, style_tensor, speed_tensor])?;

        if let Ok(output) = outputs[0].try_extract_array::<f32>() {
            let view = output.view();
            let samples = view.as_slice().unwrap().to_vec();
            let duration_seconds = samples.len() as f32 / self.sample_rate as f32;

            let audio = GeneratedAudio {
                samples,
                sample_rate: self.sample_rate,
                duration_seconds,
            };

            Ok(audio)
        } else {
            Err("Failed to extract audio output".into())
        }
    }

    #[allow(dead_code)]
    pub fn speak(
        &mut self,
        text: &str,
        voice_style: &VoiceStyle,
    ) -> Result<GeneratedAudio, Box<dyn Error>> {
        self.generate_speech(text, voice_style, 1.0)
    }
}
