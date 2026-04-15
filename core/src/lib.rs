#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::similar_names,
    clippy::missing_safety_doc
)]
#![feature(trait_alias)]

pub mod arm;
pub mod emulator;
pub mod jit;
pub mod utils;
