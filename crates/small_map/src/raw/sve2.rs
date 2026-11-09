use core::{arch::asm, mem, num::NonZeroU32, ptr};

use super::bitmask::BitMask;

#[cfg(not(all(target_arch = "aarch64", target_feature = "sve2")))]
compile_error!("The SVE2 group backend requires an AArch64 target with +sve2 enabled.");

pub(crate) type BitMaskWord = u32;
pub(crate) type NonZeroBitMaskWord = NonZeroU32;
pub(crate) const BITMASK_STRIDE: usize = 1;
pub(crate) const BITMASK_ITER_MASK: BitMaskWord = !0;

/// A fixed-size group of control bytes scanned using SVE predicates.
///
/// SVE vectors have a runtime-dependent size. Keeping a 32-byte logical
/// group allows the surrounding table to use fixed probe widths regardless
/// of whether the processor's physical vector length is 16, 32, or more bytes.
/// The bytes are copied into the group so subsequent changes to the source
/// memory do not affect the result of `match_byte`.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct Group([u8; 32]);

impl Group {
    /// Number of control bytes in a logical group.
    pub(crate) const WIDTH: usize = mem::size_of::<Self>();

    /// Masks with the lowest `n` bits set, for `0 <= n < WIDTH`.
    pub(crate) const LOWEST_MASK: [BitMaskWord; Self::WIDTH] = {
        let mut masks = [0u32; Self::WIDTH];
        let mut n = 0;
        while n < Self::WIDTH {
            masks[n] = (1u32 << n) - 1;
            n += 1;
        }
        masks
    };

    /// Copies a group of bytes from a possibly unaligned address.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid to read `Self::WIDTH` initialized bytes.
    #[inline]
    pub(crate) unsafe fn load(ptr: *const u8) -> Self {
        Self(unsafe { ptr::read_unaligned(ptr.cast::<[u8; 32]>()) })
    }

    /// Returns an exact match bitmap; bit `i` corresponds to control byte `i`.
    ///
    /// Requires an SVE2-capable CPU and operating system. This backend is
    /// compiled only when SVE2 is enabled for the whole compilation target.
    #[inline]
    pub(crate) fn match_byte(self, byte: u8) -> BitMask {
        // STR of a predicate stores one packed bit per byte lane. The largest
        // architectural SVE predicate occupies 32 bytes (2048-bit vectors).
        // Zero-init also handles cases where STR writes fewer than 4 bytes.
        let mut predicate_bytes = [0u8; 32];

        unsafe {
            asm!(
                "dup z1.b, {needle:w}",
                "cntb {vector_bytes}",
                "mov {offset}, xzr",
                "2:",
                "whilelo p0.b, {offset}, {width}",
                "ld1b {{ z0.b }}, p0/z, [{source}, {offset}]",
                "cmpeq p1.b, p0/z, z0.b, z1.b",
                "add {destination_at_offset}, {destination}, {offset}, lsr #3",
                "str p1, [{destination_at_offset}]",
                "add {offset}, {offset}, {vector_bytes}",
                "cmp {offset}, {width}",
                "b.lo 2b",
                needle = in(reg) u32::from(byte),
                source = in(reg) self.0.as_ptr(),
                destination = in(reg) predicate_bytes.as_mut_ptr(),
                width = in(reg) Self::WIDTH,
                vector_bytes = out(reg) _,
                offset = out(reg) _,
                destination_at_offset = out(reg) _,
                out("v0") _,
                out("v1") _,
                out("p0") _,
                out("p1") _,
                options(nostack),
            );
        }

        BitMask(u32::from_le_bytes([
            predicate_bytes[0],
            predicate_bytes[1],
            predicate_bytes[2],
            predicate_bytes[3],
        ]))
    }
}

#[cfg(test)]
mod tests {
    use super::{BitMaskWord, Group};

    fn expected_mask(data: &[u8; Group::WIDTH], byte: u8) -> BitMaskWord {
        let mut mask = 0;
        for (index, &value) in data.iter().enumerate() {
            if value == byte {
                mask |= 1u32 << index;
            }
        }
        mask
    }

    #[test]
    fn width_and_lowest_masks() {
        assert_eq!(Group::WIDTH, 32);
        for n in 0..Group::WIDTH {
            assert_eq!(Group::LOWEST_MASK[n], (1u32 << n) - 1);
        }
    }

    #[test]
    fn every_position_and_all_matching() {
        for position in 0..Group::WIDTH {
            let mut bytes = [0x11; Group::WIDTH];
            bytes[position] = 0xFF;
            let group = unsafe { Group::load(bytes.as_ptr()) };
            assert_eq!(group.match_byte(0xFF).0, 1u32 << position);
            assert_eq!(group.match_byte(0x11).0, !(1u32 << position));
        }

        let bytes = [0x80; Group::WIDTH];
        let group = unsafe { Group::load(bytes.as_ptr()) };
        assert_eq!(group.match_byte(0x80).0, u32::MAX);
        assert_eq!(group.match_byte(0).0, 0);
    }

    #[test]
    fn unaligned_load_and_all_256_search_values() {
        let mut data = [0u8; Group::WIDTH + 1];
        for (i, byte) in data[1..].iter_mut().enumerate() {
            *byte = ((i * 37 + 11) & 255) as u8;
        }
        let group = unsafe { Group::load(data.as_ptr().add(1)) };
        let expected_data: [u8; Group::WIDTH] = data[1..].try_into().unwrap();
        for needle in u8::MIN..=u8::MAX {
            assert_eq!(group.match_byte(needle).0, expected_mask(&expected_data, needle));
        }
    }

    #[test]
    fn load_is_a_snapshot() {
        let mut data = [7u8; Group::WIDTH];
        let group = unsafe { Group::load(data.as_ptr()) };
        data.fill(9);
        assert_eq!(group.match_byte(7).0, u32::MAX);
        assert_eq!(group.match_byte(9).0, 0);
    }
}
