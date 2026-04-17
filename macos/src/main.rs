use std::fs;

use cacao::appkit::menu::{Menu, MenuItem};
use cacao::appkit::window::{Window, WindowConfig, WindowDelegate, WindowStyle};
use cacao::appkit::{App, AppDelegate};
use cacao::color::Color;
use cacao::core_graphics;
use cacao::core_graphics::base::{kCGBitmapByteOrder16Big, kCGBitmapByteOrder32Host};
use cacao::foundation::{NSArray, NSData, NSString, NSUInteger, NSURL, id, nil};
use cacao::geometry::Rect;
use cacao::image::{Image, ImageView};
use cacao::view::View;

// Image dimensions and format (adjust to match your binary file)
const WIDTH: usize = 240;
const HEIGHT: usize = 160;
const BYTES_PER_PIXEL: usize = 3; // RGB (use 4 for RGBA)

struct PixelView {
    window: Window,
    image_view: ImageView,
    image: Image,
}

impl PixelView {
    fn new() -> Self {
        let data = vec![0u8; WIDTH * HEIGHT * BYTES_PER_PIXEL];
        let image = Image::with_data(&data);
        let image_view = ImageView::new();
        let window = Window::default();
        Self {
            window,
            image_view,
            image,
        }
    }
}

impl AppDelegate for PixelView {
    fn did_finish_launching(&self) {
        App::set_menu(vec![Menu::new(
            "",
            vec![
                MenuItem::Services,
                MenuItem::Separator,
                MenuItem::Hide,
                MenuItem::HideOthers,
                MenuItem::ShowAll,
                MenuItem::Separator,
                MenuItem::Quit,
            ],
        )]);

        App::activate();
        self.window.set_title("Hello World!");

        self.image_view.set_background_color(Color::SystemBlue);
        self.image_view.set_image(&self.image);

        self.window.show();
    }

    fn should_terminate_after_last_window_closed(&self) -> bool { true }
}

fn main() { App::new("com.example.pixelviewer", PixelView::new()).run(); }
