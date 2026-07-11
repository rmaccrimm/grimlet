use core::emulator::Emulator;
use core::emulator::video::{Pixel, SCREEN_H, SCREEN_W};
use std::time::{SystemTime, UNIX_EPOCH};

#[unsafe(no_mangle)]
pub extern "C" fn core_init() -> *mut Emulator<'static> { Box::into_raw(Box::new(Emulator::new())) }

#[unsafe(no_mangle)]
/// # Safety
/// The `emulator` pointer must be one previously returned by `core_init` and
/// not already freed
pub unsafe extern "C" fn core_drop(core_ptr: *mut Emulator<'static>) {
    unsafe {
        drop(Box::from_raw(core_ptr));
    }
}

#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
#[unsafe(no_mangle)]
/// # Safety
/// `emulator` must be a live pointer returned by `core_init`. The returned `*mut Pixel`
/// points into the emulator's internal buffer, must not be freed by the caller, and is
/// invalid after `core_drop`.
pub unsafe extern "C" fn core_get_frame_buffer(core_ptr: *mut Emulator) -> *mut Pixel {
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    let emulator = unsafe { core_ptr.as_mut_unchecked() };

    for y in 0..SCREEN_H {
        for x in 0..SCREEN_W {
            let nx = x as f64 / SCREEN_W as f64;
            let ny = y as f64 / SCREEN_H as f64;

            let wave = ((nx + ny + time) * 8.0).sin() * 0.5 + 0.5;
            let r = (nx * 255.0).round() as u8;
            let g = (ny * 255.0).round() as u8;
            let b = (wave * 255.0).round() as u8;

            emulator.frame_buffer[y * SCREEN_W + x] = Pixel { r, g, b, a: 255 };
        }
    }
    unsafe { (*core_ptr).frame_buffer.as_mut_ptr() }
}
