
#[cfg(target_arch = "x86")]
use core::arch::x86;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as x86;

use core::{mem, num::NonZeroU32};

use super::bitmask::BitMask;

pub(crate) type BitMaskWord = u32;
pub(crate) type NonZeroBitMaskWord = NonZeroU32;
pub(crate) const BITMASK_STRIDE: usize = 1;
pub(crate) const BITMASK_ITER_MASK: BitMaskWord = !0;

/// Abstraction over a group of control bytes which can be
/// scanned in parallel.
///
/// This implementation uses a 256-bit AVX2 vector.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct Group(x86::__m256i);

impl Group {
    /// Number of bytes in the group.
    pub(crate) const WIDTH: usize = mem::size_of::<Self>();

    /// Masks containing the lowest N bits set.
    pub(crate) const LOWEST_MASK: [BitMaskWord; Self::WIDTH] = {
        let mut masks = [0u32; Self::WIDTH];
        let mut i = 0;

        while i < Self::WIDTH {
            masks[i] = (1u32 << i) - 1;
            i += 1;
        }

        masks
    };

    /// Loads a group of bytes starting at the given address.
    ///
    /// The address does not need to be aligned.
    ///
    /// # Safety
    ///
    /// `ptr` must point to at least `Group::WIDTH` readable bytes.
    ///
    /// AVX2 must be enabled for the executing CPU.
    #[inline]
    #[allow(clippy::cast_ptr_alignment)]
    pub(crate) unsafe fn load(ptr: *const u8) -> Self {
        Group(unsafe { x86::_mm256_loadu_si256(ptr.cast()) })
    }

    /// Returns a `BitMask` indicating all bytes in the group
    /// which have the given value.
    ///
    /// Bit N corresponds to byte N in the group.
    #[inline]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss
    )]
    pub(crate) fn match_byte(self, byte: u8) -> BitMask {
        unsafe {
            let value = x86::_mm256_set1_epi8(byte as i8);
            let cmp = x86::_mm256_cmpeq_epi8(self.0, value);
            let mask = x86::_mm256_movemask_epi8(cmp);

            BitMask(mask as u32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BitMaskWord, Group};

    #[test]
    fn test_width() {
        assert_eq!(Group::WIDTH, 32);
    }

    #[test]
    fn test_lowest_masks() {
        for i in 0..Group::WIDTH {
            assert_eq!(
                Group::LOWEST_MASK[i],
                (1u32 << i) - 1
            );
        }
    }

    #[test]
    fn test_all_bytes_match() {
        let bytes = [0x42u8; Group::WIDTH];

        let group = unsafe { Group::load(bytes.as_ptr()) };

        assert_eq!(
            group.match_byte(0x42).0,
            BitMaskWord::MAX
        );

        assert_eq!(group.match_byte(0x43).0, 0);
    }

    #[test]
    fn test_every_byte_position() {
        for position in 0..Group::WIDTH {
            let mut bytes = [0u8; Group::WIDTH];
            bytes[position] = 0x42;

            let group = unsafe { Group::load(bytes.as_ptr()) };

            assert_eq!(
                group.match_byte(0x42).0,
                1u32 << position
            );
        }
    }

    #[test]
    fn test_unaligned_load_and_all_byte_values() {
        let mut buffer = [0u8; Group::WIDTH + 1];

        for (i, byte) in buffer[1..].iter_mut().enumerate() {
            *byte = ((i * 37 + 11) & 0xff) as u8;
        }

        let ptr = unsafe { buffer.as_ptr().add(1) };
        let group = unsafe { Group::load(ptr) };

        for target in u8::MIN..=u8::MAX {
            let mut expected = 0u32;

            for (i, &value) in buffer[1..].iter().enumerate() {
                if value == target {
                    expected |= 1u32 << i;
                }
            }

            assert_eq!(
                group.match_byte(target).0,
                expected,
                "Mismatch for byte {target}"
            );
        }
    }
}
