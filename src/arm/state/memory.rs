use std::array::TryFromSliceError;
use std::slice::Chunks;

use anyhow::{Result, bail};
use num::traits::{AsPrimitive, FromBytes, ToBytes};

#[repr(C)]
pub struct MainMemory {
    bios: Vec<u8>,
    slow_wram: Vec<u8>,
    fast_wram: Vec<u8>,
    io_registers: Vec<u8>,
    palette_ram: Vec<u8>,
    vram: Vec<u8>,
    obj_attrs: Vec<u8>,
    cartridge_rom: Vec<u8>,
    io_regs: IORegisters,
}

/// Memory-Mapped IO Registers. Padding fields allow for easier direct memory access
#[repr(C, packed)]
#[derive(Default)]
pub struct IORegisters {
    // Display registers (0x04000000 - 0x04000054)
    pub dispcnt: u16,
    pub stereo_enable: u16,
    pub dispstat: u16,
    pub vcount: u16,
    pub bg0cnt: u16,
    pub bg1cnt: u16,
    pub bg2cnt: u16,
    pub bg3cnt: u16,
    pub bg0hofs: u16,
    pub bg0vofs: u16,
    pub bg1hofs: u16,
    pub bg1vofs: u16,
    pub bg2hofs: u16,
    pub bg2vofs: u16,
    pub bg3hofs: u16,
    pub bg3vofs: u16,
    pub bg2pa: u16,
    pub bg2pb: u16,
    pub bg2pc: u16,
    pub bg2pd: u16,
    pub bg2x: u32,
    pub bg2y: u32,
    pub bg3pa: u16,
    pub bg3pb: u16,
    pub bg3pc: u16,
    pub bg3pd: u16,
    pub bg3x: u32,
    pub bg3y: u32,
    pub win0h: u16,
    pub win1h: u16,
    pub win0v: u16,
    pub win1v: u16,
    pub winout: u16,
    pub mosaic: u16,
    pub __padding1: u16,
    pub bldcnt: u16,
    pub bldalpha: u16,
    pub bldy: u16,
    pub __padding2: [u8; 10],

    // Sound registers (0x04000060 - 0x040000A4)
    pub sound1cnt_l: u16,
    pub sound1cnt_h: u16,
    pub sound1cnt_x: u16,
    pub __padding3: u16,
    pub sound2cnt_l: u16,
    pub __padding4: u16,
    pub sound2cnt_h: u16,
    pub __padding5: u16,
    pub sound3cnt_l: u16,
    pub sound3cnt_h: u16,
    pub sound3cnt_x: u16,
    pub __padding6: u16,
    pub sound4cnt_l: u16,
    pub __padding7: u16,
    pub sound4cnt_h: u16,
    pub __padding8: u16,
    pub soundcnt_l: u16,
    pub soundcnt_h: u16,
    pub soundcnt_x: u16,
    pub __padding9: u16,
    pub soundbias: u16,
    pub __padding10: [u8; 22],
    pub fifo_a: u32,
    pub fifo_b: u32,
    pub __padding11: [u8; 8],

    // DMA registers (0x040000B0 - 0x040000DE)
    pub dma0sad: u32,
    pub dma0dad: u32,
    pub dma0cnt_l: u16,
    pub dma0cnt_h: u16,
    pub dma1sad: u32,
    pub dma1dad: u32,
    pub dma1cnt_l: u16,
    pub dma1cnt_h: u16,
    pub dma2sad: u32,
    pub dma2dad: u32,
    pub dma2cnt_l: u16,
    pub dma2cnt_h: u16,
    pub dma3sad: u32,
    pub dma3dad: u32,
    pub dma3cnt_l: u16,
    pub dma3cnt_h: u16,
    pub __padding12: [u8; 32],

    // Timer registers (0x04000100 - 0x0400010E)
    pub tm0d: u16,
    pub tm0cnt: u16,
    pub tm1d: u16,
    pub tm1cnt: u16,
    pub tm2d: u16,
    pub tm2cnt: u16,
    pub tm3d: u16,
    pub tm3cnt: u16,
    pub __padding13: [u8; 16],

    // Serial I/O registers (0x04000120 - 0x0400012A)
    pub scd0: u16,
    pub scd1: u16,
    pub scd2: u16,
    pub scd3: u16,
    pub sccnt_l: u16,
    pub sccnt_h: u16,
    pub __padding14: [u8; 4],
    pub keyinput: u16,
    pub keycnt: u16,
    pub rcnt: u16,
    pub __padding15: [u64; 25],
    pub __padding16: u16,

    // Interrupt/System registers (0x04000200 - 0x04000300)
    pub ie: u16,
    pub if_reg: u16,
    pub waitcnt: u16,
    pub __padding17: u16,
    pub ime: u16,
    pub __padding18: [u64; 30],
    pub __padding19: [u8; 6],
    pub haltcnt: u8,
}

pub trait MemReadable =
    AsPrimitive<u32> + for<'a> FromBytes<Bytes: TryFrom<&'a [u8], Error = TryFromSliceError>>;

pub trait MemWriteable = ToBytes<Bytes: IntoIterator<Item = u8>>;

impl MainMemory {
    pub fn iter_word(&self, start_addr: u32) -> Result<Chunks<'_, u8>> {
        if !start_addr.is_multiple_of(4) {
            panic!("Mis-alligned word address: {:x}", start_addr);
        }
        Ok(self.mem_map_lookup(start_addr)?.chunks(4))
    }

    pub fn iter_halfword(&self, start_addr: u32) -> Result<Chunks<'_, u8>> {
        if !start_addr.is_multiple_of(2) {
            panic!("Mis-alligned halfword address: {:x}", start_addr);
        }
        Ok(self.mem_map_lookup(start_addr)?.chunks(2))
    }

    /// Sign or zero-extends the result to 32 bits depending type parameter
    pub fn read<T>(&self, addr: u32) -> u32
    where
        T: MemReadable,
    {
        let mem_slice = self
            .mem_map_lookup(addr)
            .expect("out of bounds read")
            .get(0..size_of::<T>())
            .expect("reached end of bytes while reading");
        let bytes: T::Bytes = mem_slice.try_into().expect("conversion from bytes failed");
        let as_int = T::from_le_bytes(&bytes);
        as_int.as_()
    }

    pub fn write<T>(&mut self, addr: u32, value: T)
    where
        T: MemWriteable,
    {
        let mut mem_iter = self
            .mem_map_lookup_mut(addr)
            .expect("out of bounds write")
            .iter_mut();
        for byte in value.to_le_bytes() {
            let mem_val = mem_iter.next().expect("reached end of bytes while writing");
            *mem_val = byte;
        }
    }

    pub fn get_ptr(&self, addr: u32) -> *const u8 {
        self.mem_map_lookup(addr)
            .expect("get_ptr address out of bounds")
            .as_ptr()
    }

    /// Planning to use this to be slightly more convenient for multiple stores. TODO what if
    /// they were writing to memory mapped IO? Could that happen?
    pub fn get_mut_ptr(&mut self, addr: u32) -> *mut u8 {
        self.mem_map_lookup_mut(addr)
            .expect("get_mut_ptr address out of bounds")
            .as_mut_ptr()
    }

    pub fn mem_map_lookup(&self, addr: u32) -> Result<&[u8]> {
        let base = addr & 0x0f000000;
        let index = (addr - base) as usize;
        let region = match base {
            0x00000000 => &self.bios,
            0x02000000 => &self.slow_wram,
            0x03000000 => &self.fast_wram,
            0x04000000 => &self.io_registers,
            0x05000000 => &self.palette_ram,
            0x06000000 => &self.vram,
            0x07000000 => &self.obj_attrs,
            0x08000000 => &self.cartridge_rom,
            _ => bail!("unused area of memory (addr: {:#08x})", addr),
        };
        Ok(&region[index..])
    }

    // Would be nice if these could be combined somehow
    pub fn mem_map_lookup_mut(&mut self, addr: u32) -> Result<&mut [u8]> {
        let base = addr & 0x0f000000;
        let index = (addr - base) as usize;
        let region = match base {
            0x00000000 => &mut self.bios,
            0x02000000 => &mut self.slow_wram,
            0x03000000 => &mut self.fast_wram,
            0x04000000 => &mut self.io_registers,
            0x05000000 => &mut self.palette_ram,
            0x06000000 => &mut self.vram,
            0x07000000 => &mut self.obj_attrs,
            0x08000000 => &mut self.cartridge_rom,
            _ => bail!("unused area of memory (addr: {:#08x})", addr),
        };
        Ok(&mut region[index..])
    }
}

impl Default for MainMemory {
    fn default() -> Self {
        Self {
            bios: vec![0; 16 << 10],
            slow_wram: vec![0; 256 << 10],
            fast_wram: vec![0; 32 << 10],
            io_registers: vec![0; 0x3ff],
            palette_ram: vec![0; 1 << 10],
            vram: vec![0; 96 << 10],
            obj_attrs: vec![0; 1 << 10],
            cartridge_rom: vec![0; 32 << 20],
            io_regs: IORegisters::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::arm::state::memory::MainMemory;

    macro_rules! read_tests {
        ($bytes:expr, $($name:ident: $T:ty, $data:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    let (read_addr, expected) = $data;

                    let mut mem = MainMemory::default();
                    mem.bios = $bytes;
                    assert_eq!(mem.read::<$T>(read_addr), expected);
                }
            )*
        };
    }

    read_tests! {
        vec![0x34, 0xff, 0xbe, 0x70, 0xf1],
        test_read_unsigned_0: u8, (0, 0x00000034),
        test_read_unsigned_1: u8, (1, 0x000000ff),
        test_read_unsigned_2: u8, (2, 0x000000be),
        test_read_unsigned_3: u16, (0, 0x0000ff34),
        test_read_unsigned_4: u16, (1, 0x0000beff),
        test_read_unsigned_5: u16, (2, 0x000070be),
        test_read_unsigned_6: u32, (0, 0x70beff34),
        test_read_unsigned_7: u32, (1, 0xf170beff),
    }

    read_tests! {
        vec![0x84, 0xff, 0x3e, 0x70, 0x80],
        test_read_signed_0: i8, (0, 0xffffff84),
        test_read_signed_1: i8, (2, 0x0000003e),
        test_read_signed_2: i16, (0, 0xffffff84),
        test_read_signed_3: i16, (1, 0x00003eff),
        test_read_signed_4: i32, (0, 0x703eff84),
        test_read_signed_5: i32, (1, 0x80703eff),
    }

    #[test]
    #[should_panic = "reached end of bytes while reading"]
    fn test_read_past_end_of_bytes_panics() {
        let mem = MainMemory {
            bios: vec![0x34, 0xff, 0xbe, 0x70],
            ..Default::default()
        };
        mem.read::<u32>(1);
    }

    #[test]
    fn test_write() {
        let mut mem = MainMemory {
            bios: vec![0; 4],
            ..Default::default()
        };
        mem.write(0, 0x12345678u32);
        assert_eq!(mem.bios, vec![0x78, 0x56, 0x34, 0x12]);
        mem.write(0, 0u8);
        assert_eq!(mem.bios, vec![0x00, 0x56, 0x34, 0x12]);
        mem.write(1, 0u16);
        assert_eq!(mem.bios, vec![0x00, 0x00, 0x00, 0x12]);
        mem.write(0, 0x3102u16);
        assert_eq!(mem.bios, vec![0x02, 0x31, 0x00, 0x12]);
        mem.write(0, -12i8);
        assert_eq!(mem.bios, vec![0xf4, 0x31, 0x00, 0x12]);
        mem.write(2, 0xabu8);
        assert_eq!(mem.bios, vec![0xf4, 0x31, 0xab, 0x12]);
        mem.write(0, -10500i16);
        assert_eq!(mem.bios, vec![0xfc, 0xd6, 0xab, 0x12]);
    }
}
