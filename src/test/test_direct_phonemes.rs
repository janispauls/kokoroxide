use ort::session::builder::GraphOptimizationLevel;

use crate::kokoro::{load_voice_style, KokoroTTS, TTSConfig};
use crate::playback::play_wav_file;

pub fn test_direct_phonemes() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Direct Phoneme Input ===\n");

    // Initialize TTS
    let tts_config = TTSConfig::new("models/kokoro/kokoro.onnx", "models/kokoro/tokenizer.json")
        .with_graph_optimization_level(GraphOptimizationLevel::Disable)
        .with_max_tokens_length(512)
        .with_sample_rate(24000);

    let mut tts = KokoroTTS::with_config(tts_config)?;

    let voice_style = load_voice_style("models/kokoro/af.bin")?;

    // Test cases with direct phonemes
    let test_cases = vec![
        (
            "Kokoro g2p - Japanese anime",
            "ʤˌæpənˈiz klˈʌbz ɑɹ ðə bˈɛst plˈʌs ˈænəmA ɪz əmˈAzɪŋ",
        ),
        (
            "Our system - Japanese anime",
            "ʤˌæpənˈiz klˈʌbz ɑɹ ðə bˈɛst plˈʌs ˈænɪmˌA ɪz ɐmˈAzɪŋ",
        ),
        (
            "Our system phonemes (short)",
            "ɪt wʌzə spɹˈɔl vˈYs ænd ə spɹˈɔl ʤˈOk",
        ),
        (
            "Kokoro Python phonemes (short)",
            "ˌɪt wʌz ɐ spɹˈɔl vˈYs ænd ɐ spɹˈɔl ʤˈOk",
        ),
    ];

    for (description, phonemes) in test_cases {
        println!("Test: {}", description);
        println!("Input phonemes: '{}'", phonemes);

        // Generate speech from phonemes directly
        let filename = format!(
            "test_phonemes_{}.wav",
            description
                .to_lowercase()
                .replace(" ", "_")
                .replace("-", "_")
        );

        println!("Generating speech to: {}", filename);

        match tts.generate_speech_from_phonemes(phonemes, &voice_style, 1.0) {
            Ok(audio) => {
                println!("✓ Generated successfully");
                println!("  Duration: {:.2}s", audio.duration_seconds);

                // Save the audio file
                audio.save_to_wav(&filename)?;

                // Play the audio
                std::thread::sleep(std::time::Duration::from_millis(500));
                if let Err(e) = play_wav_file(&filename) {
                    eprintln!("  Playback error: {}", e);
                }
            }
            Err(e) => eprintln!("✗ Generation failed: {}", e),
        }

        println!();
    }

    Ok(())
}

#[allow(dead_code)]
pub fn run_phoneme_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Phoneme Comparison Test ===\n");

    let test_text = "It was a Sprawl voice and a Sprawl joke. The Chatsubo was a bar for professional expatriates; you could drink there for a week and never hear two words in Japanese";

    println!("Test text: '{}'\n", test_text);

    // Our system's phonemes
    let our_phonemes = "ɪt wʌzə spɹˈɔl vˈYs ænd ə spɹˈɔl ʤˈOk ðə ʧætsˈubO wʌzə bˈɑɹ fɔɹ pɹəfˈɛʃənəl ɛkspˈAtɹɪˌAts ju kʊd dɹˈɪŋk ðɛɹ fəɹɹə wˈik ænd nˈɛvəɹ hˈɪɹ tˈu wˈɜɹdz ɪn ʤˌæpənˈiz";

    // Kokoro Python's phonemes
    let kokoro_phonemes = "ˌɪt wʌz ɐ spɹˈɔl vˈYs ænd ɐ spɹˈɔl ʤˈOk. ðə ʧætsˈubO wʌz ɐ bˈɑɹ fɔɹ pɹəfˈɛʃᵊnəl ɛkspˈAtɹiəts; ju kʊd dɹˈɪŋk ðɛɹ fɔɹ ɐ wˈik ænd nˈɛvəɹ hˈɪɹ tˈu wˈɜɹdz ɪn ʤˌæpənˈiz.";

    println!("Our phonemes:\n{}\n", our_phonemes);
    println!("Kokoro phonemes:\n{}\n", kokoro_phonemes);

    // Analyze differences
    println!("Key differences:");
    println!("1. Schwa representation: 'ə' vs 'ɐ'");
    println!("2. Syllabic consonants: 'ənəl' vs 'ᵊnəl'");
    println!("3. Length markers: present vs absent (ː)");
    println!("4. Stress markers: missing initial 'ˌ'");
    println!("5. Punctuation: stripped vs preserved");
    println!("6. Diphthongs: 'eɪ' vs already converted 'A'");

    Ok(())
}
