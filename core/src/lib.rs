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

use emulator::Emulator;

#[unsafe(no_mangle)]
pub extern "C" fn core_emulator_new() -> *mut Emulator<'static> {
    Box::into_raw(Box::new(Emulator::new()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn core_emulator_next_frame(e: *mut Emulator<'static>) -> *mut u8 {
    let emulator = unsafe { e.as_mut().expect("null pointer to emulator") };
    emulator.run(|_| false);
    emulator.next_frame.as_mut_ptr()
}
