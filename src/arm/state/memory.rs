use std::array::TryFromSliceError;
use std::cell::RefCell;
use std::rc::Rc;
use std::slice::Chunks;
use std::sync::mpsc::Sender;

use anyhow::{Result, bail};
use num::traits::{AsPrimitive, FromBytes, ToBytes};

use crate::utils::interval_tree::IntervalTree;

// TODO - will add these as needed
pub enum IoReg {
    DISPCNT = 0x0400_0000,
    DISPSTAT = 0x0400_0004,
}

/// Always returned as a u32 because we need to know it's width within the JIT code
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ReadVal {
    pub value: u32,
    pub wait_states: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct WriteVal {
    pub should_exit: bool,
    pub wait_states: u32,
}

#[derive(Clone, Debug)]
struct MemRegion {
    wait_states: u32,
    data: Vec<u8>,
}

impl MemRegion {
    fn new(size: usize, wait_states: u32) -> Self {
        MemRegion {
            wait_states,
            data: vec![0; size],
        }
    }
}

pub trait MemReadable = AsPrimitive<u32>
    + for<'a> FromBytes<Bytes: TryFrom<&'a [u8], Error = TryFromSliceError>>
    + Copy;

pub trait MemWriteable = ToBytes<Bytes: IntoIterator<Item = u8>> + Copy;

pub struct MemoryManager {
    bios: MemRegion,
    external_wram: MemRegion,
    internal_wram: MemRegion,
    io_registers: MemRegion,
    palette_ram: MemRegion,
    vram: MemRegion,
    obj_attrs: MemRegion,
    cartridge_rom: MemRegion,

    current_addr_range: (u32, u32),

    // Both of these are optional to make testing more straightforward, may rework this later.
    // Shared with `FunctionCache`
    interval_tree: Option<Rc<RefCell<IntervalTree<u32>>>>,
    tx: Option<Sender<u32>>,
}

impl MemoryManager {
    pub fn new(interval_tree: Rc<RefCell<IntervalTree<u32>>>, tx: Sender<u32>) -> Self {
        Self {
            interval_tree: Some(interval_tree),
            tx: Some(tx),
            ..Self::default()
        }
    }

    pub fn set_curr_addr_range(&mut self, start: u32, end: u32) {
        self.current_addr_range = (start, end);
    }

    pub fn iter_word(&self, start_addr: u32) -> Result<Chunks<'_, u8>> {
        assert!(
            start_addr.is_multiple_of(4),
            "Mis-alligned word address: {start_addr:x}"
        );
        Ok(self.mem_map_lookup(start_addr)?.0.chunks(4))
    }

    pub fn iter_halfword(&self, start_addr: u32) -> Result<Chunks<'_, u8>> {
        assert!(
            start_addr.is_multiple_of(2),
            "Mis-alligned halfword address: {start_addr:x}"
        );
        Ok(self.mem_map_lookup(start_addr)?.0.chunks(2))
    }

    /// Sign or zero-extends the result to 32 bits depending type parameter
    /// Returns both the read value and the number of wait-states
    ///
    /// TODO - I think we need the current PC here to correctly determine delay. Maybe that can
    /// be done from the call site instead.
    pub fn read<T>(&self, addr: u32) -> ReadVal
    where
        T: MemReadable,
    {
        let (mem_slice, wait_states) = self.mem_map_lookup(addr).expect("out of bounds read");
        let bytes: T::Bytes = mem_slice
            .get(0..size_of::<T>())
            .expect("reached end of bytes while reading")
            .try_into()
            .expect("conversion from bytes failed");
        let as_int = T::from_le_bytes(&bytes);
        ReadVal {
            value: as_int.as_(),
            wait_states,
        }
    }

    /// Writes can potentially invalidate a code block, including the one currently running, in
    /// which case we need to exit the running block.
    pub fn write<T>(&mut self, addr: u32, value: T) -> WriteVal
    where
        T: MemWriteable,
    {
        let (mem, wait_states) = self.mem_map_lookup_mut(addr).expect("out of bounds write");
        let mut mem_iter = mem.iter_mut();
        for byte in value.to_le_bytes() {
            let mem_val = mem_iter.next().expect("reached end of bytes while writing");
            *mem_val = byte;
        }
        let mut should_exit = false;
        // Perform cache invalidation - removes any nodes in the interval tree that contain the
        // written address and sends a message to FunctionCache to do the same (we cannot do this
        // immediately as the currently executing function may be invalidated).
        if let Some(t) = &self.interval_tree {
            let removed = t.borrow_mut().remove_all(addr);
            if let Some(tx) = &self.tx {
                for r in removed {
                    if r == self.current_addr_range {
                        should_exit = true;
                    }
                    tx.send(r.0).expect("cache <-> memory channel was closed");
                }
            }
        }
        WriteVal {
            should_exit,
            wait_states,
        }
    }

    // TODO - should these be public?
    pub fn mem_map_lookup(&self, addr: u32) -> Result<(&[u8], u32)> {
        let base = addr & 0x0f00_0000;
        let index = (addr - base) as usize;
        let region = match base {
            0x0000_0000 => &self.bios,
            0x0200_0000 => &self.external_wram,
            0x0300_0000 => &self.internal_wram,
            0x0400_0000 => &self.io_registers,
            0x0500_0000 => &self.palette_ram,
            0x0600_0000 => &self.vram,
            0x0700_0000 => &self.obj_attrs,
            0x0800_0000 => &self.cartridge_rom,
            _ => bail!("unused area of memory (addr: {addr:#08x})"),
        };
        Ok((&region.data[index..], region.wait_states))
    }

    // Would be nice if these could be combined somehow
    pub fn mem_map_lookup_mut(&mut self, addr: u32) -> Result<(&mut [u8], u32)> {
        let base = addr & 0x0f00_0000;
        let index = (addr - base) as usize;
        let region = match base {
            0x0000_0000 => &mut self.bios,
            0x0200_0000 => &mut self.external_wram,
            0x0300_0000 => &mut self.internal_wram,
            0x0400_0000 => &mut self.io_registers,
            0x0500_0000 => &mut self.palette_ram,
            0x0600_0000 => &mut self.vram,
            0x0700_0000 => &mut self.obj_attrs,
            0x0800_0000 => &mut self.cartridge_rom,
            _ => bail!("unused area of memory (addr: {addr:#08x})"),
        };
        Ok((&mut region.data[index..], region.wait_states))
    }

    // Returns (IO register, wait-states).
    // Not all IO registers are actually 32-bits wide. Leave that up to the caller
    pub fn read_io(&self, reg: IoReg) -> ReadVal { self.read::<u32>(reg as u32) }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self {
            bios: MemRegion::new(16 << 10, 0),
            external_wram: MemRegion::new(256 << 10, 0),
            internal_wram: MemRegion::new(32 << 10, 0),
            io_registers: MemRegion::new(0x400, 0),
            palette_ram: MemRegion::new(1 << 10, 0),
            vram: MemRegion::new(96 << 10, 0),
            obj_attrs: MemRegion::new(1 << 10, 0),
            // TODO -  wait states configurable? Possibly seperate struct, maybe a Readable trait?
            cartridge_rom: MemRegion::new(32 << 20, 0),
            current_addr_range: (0, 0),
            interval_tree: None,
            tx: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! read_tests {
        ($bytes:expr, $($name:ident: $T:ty, $data:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    let (read_addr, expected) = $data;

                    let mut mem = MemoryManager::default();
                    mem.bios = MemRegion{data: $bytes, wait_states: 0};
                    assert_eq!(mem.read::<$T>(read_addr).value, expected);
                }
            )*
        };
    }

    read_tests! {
        vec![0x34, 0xff, 0xbe, 0x70, 0xf1],
        test_read_unsigned_0: u8, (0, 0x0000_0034),
        test_read_unsigned_1: u8, (1, 0x0000_00ff),
        test_read_unsigned_2: u8, (2, 0x0000_00be),
        test_read_unsigned_3: u16, (0, 0x0000_ff34),
        test_read_unsigned_4: u16, (1, 0x0000_beff),
        test_read_unsigned_5: u16, (2, 0x0000_70be),
        test_read_unsigned_6: u32, (0, 0x70be_ff34),
        test_read_unsigned_7: u32, (1, 0xf170_beff),
    }

    read_tests! {
        vec![0x84, 0xff, 0x3e, 0x70, 0x80],
        test_read_signed_0: i8, (0, 0xffff_ff84),
        test_read_signed_1: i8, (2, 0x0000_003e),
        test_read_signed_2: i16, (0, 0xfff_fff84),
        test_read_signed_3: i16, (1, 0x0000_3eff),
        test_read_signed_4: i32, (0, 0x703e_ff84),
        test_read_signed_5: i32, (1, 0x8070_3eff),
    }

    #[test]
    #[should_panic = "reached end of bytes while reading"]
    fn test_read_past_end_of_bytes_panics() {
        let mem = MemoryManager {
            bios: MemRegion {
                data: vec![0x34, 0xff, 0xbe, 0x70],
                wait_states: 0,
            },
            ..Default::default()
        };
        mem.read::<u32>(1);
    }

    #[test]
    fn test_write() {
        let mut mem = MemoryManager {
            bios: MemRegion::new(4, 0),
            ..Default::default()
        };
        mem.write(0, 0x1234_5678);
        assert_eq!(mem.bios.data, vec![0x78, 0x56, 0x34, 0x12]);
        mem.write(0, 0u8);
        assert_eq!(mem.bios.data, vec![0x00, 0x56, 0x34, 0x12]);
        mem.write(1, 0u16);
        assert_eq!(mem.bios.data, vec![0x00, 0x00, 0x00, 0x12]);
        mem.write(0, 0x3102u16);
        assert_eq!(mem.bios.data, vec![0x02, 0x31, 0x00, 0x12]);
        mem.write(0, -12i8);
        assert_eq!(mem.bios.data, vec![0xf4, 0x31, 0x00, 0x12]);
        mem.write(2, 0xabu8);
        assert_eq!(mem.bios.data, vec![0xf4, 0x31, 0xab, 0x12]);
        mem.write(0, -10500i16);
        assert_eq!(mem.bios.data, vec![0xfc, 0xd6, 0xab, 0x12]);
    }
}
