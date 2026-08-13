// src/main.rs — CLI entry point for fleet-jepa-midi

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use fleet_jepa_midi::{
    director::Director,
    engine::EngineRegistry,
    jepa::JepaEncoder,
};

#[derive(Parser)]
#[command(name = "fleet-jepa-midi")]
#[command(about = "Three-layer real-time music intelligence")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate MIDI from an engine
    Generate {
        #[arg(long)]
        engine: String,
        #[arg(long, default_value = "16")]
        bars: usize,
        #[arg(long, default_value = "output.mid")]
        output: String,
    },
    /// Compute JEPA embedding for a MIDI file
    Embed {
        #[arg(long)]
        input: String,
    },
    /// Run the director (connects to fleet-gateway)
    Run {
        #[arg(long, default_value = "http://localhost:3000")]
        gateway: String,
    },
    /// Run the self-improving harness
    Harness {
        #[arg(long, default_value = "100")]
        iterations: usize,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Generate { engine, bars, output }) => {
            tracing::info!("generating {bars} bars with engine '{engine}'");
            let registry = EngineRegistry::new();
            let gen = registry.generate(&engine, bars)?;
            tracing::info!("generated {} notes", gen.len());
            // TODO: write to MIDI file
            tracing::info!("would write to {output}");
        }
        Some(Commands::Embed { input }) => {
            tracing::info!("embedding MIDI file: {input}");
            let data = std::fs::read(&input)?;
            let smf = midly::Smf::parse(&data)?;
            let encoder = JepaEncoder::new();
            let bars = fleet_jepa_midi::midi::smf_to_bars(&smf);
            for (i, bar) in bars.iter().enumerate() {
                let emb = encoder.embed_bar(bar);
                tracing::info!("bar {i}: {:?}", emb);
            }
        }
        Some(Commands::Run { gateway }) => {
            tracing::info!("starting director, gateway={gateway}");
            let mut director = Director::new(&gateway);
            director.run().await?;
        }
        Some(Commands::Harness { iterations }) => {
            tracing::info!("running harness for {iterations} iterations");
            let mut harness = fleet_jepa_midi::harness::CuriosityHarness::new();
            harness.run(iterations);
            tracing::info!("harness complete. best score: {:.4}", harness.best_score());
        }
        None => {
            tracing::info!("fleet-jepa-midi — no command given. Try --help.");
        }
    }

    Ok(())
}
