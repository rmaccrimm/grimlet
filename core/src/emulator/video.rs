#![allow(dead_code)]
pub struct Sprite<'a> {
    mem: &'a [u8],
}

enum ObjMode {}

struct ObjShape {
    width: usize,
    height: usize,
}

enum ColorMode {}

impl Sprite<'_> {
    fn y(&self) -> u8 { todo!() }

    fn rotate_scale(&self) -> bool { todo!() }

    fn double_disable(&self) -> bool { todo!() }

    fn obj_mode(&self) -> ObjMode { todo!() }

    fn mosaic(&self) -> bool { todo!() }

    fn color_mode(&self) -> ColorMode { todo!() }

    fn shape(&self) -> ObjShape { todo!() }

    fn x(&self) -> u16 { todo!() }

    fn rotate_scale_group(&self) -> u8 { todo!() }

    fn horizontal_flip(&self) -> bool { todo!() }

    fn vertical_flip(&self) -> bool { todo!() }

    fn obj_size(&self) -> u8 { todo!() }

    fn character_name(&self) -> u16 { todo!() }

    fn priority(&self) -> u8 { todo!() }

    fn palette_number(&self) -> u8 { todo!() }
}
