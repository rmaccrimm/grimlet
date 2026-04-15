pub mod interval_tree;

// Reinterpret bits and sign-extend
pub trait CastToU64 {
    fn cast_to_u64(self) -> u64;
}

impl CastToU64 for u8 {
    fn cast_to_u64(self) -> u64 { u64::from(self) }
}

impl CastToU64 for u16 {
    fn cast_to_u64(self) -> u64 { u64::from(self) }
}

impl CastToU64 for u32 {
    fn cast_to_u64(self) -> u64 { u64::from(self) }
}

impl CastToU64 for u64 {
    fn cast_to_u64(self) -> u64 { self }
}

impl CastToU64 for usize {
    fn cast_to_u64(self) -> u64 { u64::try_from(self).expect("cast from usize to u64 failed") }
}

impl CastToU64 for i8 {
    fn cast_to_u64(self) -> u64 { u64::from(self.cast_unsigned()) }
}

impl CastToU64 for i16 {
    fn cast_to_u64(self) -> u64 { u64::from(self.cast_unsigned()) }
}

impl CastToU64 for i32 {
    fn cast_to_u64(self) -> u64 { u64::from(self.cast_unsigned()) }
}

impl CastToU64 for i64 {
    fn cast_to_u64(self) -> u64 { self.cast_unsigned() }
}

impl CastToU64 for isize {
    fn cast_to_u64(self) -> u64 {
        u64::try_from(self.cast_unsigned()).expect("cast from isize to u64 failed")
    }
}
