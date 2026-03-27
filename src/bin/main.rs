#![deny(clippy::all)]
#![forbid(unsafe_code)]

use eframe::egui::{self, Color32, ColorImage, TextureOptions};

const WIDTH: usize = 320;
const HEIGHT: usize = 240;
const BOX_SIZE: i16 = 64;

/// Representation of the application state. In this example, a box will bounce around the screen.
struct World {
    box_x: i16,
    box_y: i16,
    velocity_x: i16,
    velocity_y: i16,
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([WIDTH as f32, HEIGHT as f32]),
        ..Default::default()
    };
    eframe::run_native(
        "Image Viewer",
        options,
        Box::new(|_| Ok(Box::new(MyApp::new()))),
    )
}

struct MyApp {
    world: World,
    img: ColorImage,
}

impl MyApp {
    fn new() -> Self {
        Self {
            world: World::new(),
            img: ColorImage::filled([WIDTH, HEIGHT], Color32::LIGHT_BLUE),
        }
    }
}

impl eframe::App for MyApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::CentralPanel::no_frame().show_inside(ui, |ui| {
            let tex = ui.load_texture("img", self.img.clone(), TextureOptions::default());
            ui.image((tex.id(), tex.size_vec2()));
        });

        self.world.update();
        self.img = self.world.draw();
        ui.request_repaint();
    }
}

impl World {
    /// Create a new `World` instance that can draw a moving box.
    fn new() -> Self {
        Self {
            box_x: 24,
            box_y: 16,
            velocity_x: 1,
            velocity_y: 1,
        }
    }

    /// Update the `World` internal state; bounce the box around the screen.
    fn update(&mut self) {
        if self.box_x <= 0 || self.box_x + BOX_SIZE > WIDTH as i16 {
            self.velocity_x *= -1;
        }
        if self.box_y <= 0 || self.box_y + BOX_SIZE > HEIGHT as i16 {
            self.velocity_y *= -1;
        }

        self.box_x += self.velocity_x;
        self.box_y += self.velocity_y;
    }

    /// Draw the `World` state to the frame buffer.
    ///
    /// Assumes the default texture format: `wgpu::TextureFormat::Rgba8UnormSrgb`
    fn draw(&self) -> ColorImage {
        let mut buffer: Vec<Color32> = vec![Color32::LIGHT_BLUE; WIDTH * HEIGHT];

        for (i, pixel) in buffer.iter_mut().enumerate() {
            let x = (i % WIDTH) as i16;
            let y = (i / WIDTH) as i16;

            let inside_the_box = x >= self.box_x
                && x < self.box_x + BOX_SIZE
                && y >= self.box_y
                && y < self.box_y + BOX_SIZE;

            if inside_the_box {
                *pixel = Color32::BLUE;
            }
        }

        ColorImage::new([WIDTH, HEIGHT], buffer)
    }
}
