use super::imp::{BitMaskWord, NonZeroBitMaskWord, BITMASK_ITER_MASK, BITMASK_STRIDE};

/// A bit mask which contains the result of a `Match` operation on a `Group` and
/// allows iterating through them.
///
/// The bit mask is arranged so that low-order bits represent lower memory
/// addresses for group match results.
///
/// For implementation reasons, the bits in the set may be sparsely packed with
/// groups of 8 bits representing one element. If any of these bits are non-zero
/// then this element is considered to true in the mask. If this is the
/// case, `BITMASK_STRIDE` will be 8 to indicate a divide-by-8 should be
/// performed on counts/indices to normalize this difference. The backend's
/// `BITMASK_ITER_MASK` selects the used bit in each logical element.
///
/// To iterate over a bit mask, it must be converted to a form where only 1 bit
/// is set per element. This is done by applying `BITMASK_ITER_MASK` on the
/// mask bits.
#[derive(Copy, Clone)]
pub(crate) struct BitMask(pub(crate) BitMaskWord);

#[allow(clippy::use_self)]
impl BitMask {
    #[inline]
    pub(crate) fn and(self, n: BitMaskWord) -> Self {
        Self(self.0 & n)
    }

    /// Returns a new `BitMask` with the lowest bit removed.
    #[inline]
    #[must_use]
    fn remove_lowest_bit(self) -> Self {
        BitMask(self.0 & (self.0 - 1))
    }

    /// Returns the first set bit in the `BitMask`, if there is one.
    #[inline]
    pub(crate) fn lowest_set_bit(self) -> Option<usize> {
        if let Some(nonzero) = NonZeroBitMaskWord::new(self.0) {
            Some(Self::nonzero_trailing_zeros(nonzero))
        } else {
            None
        }
    }

    /// Same as above but takes a `NonZeroBitMaskWord`.
    #[inline]
    fn nonzero_trailing_zeros(nonzero: NonZeroBitMaskWord) -> usize {
        if cfg!(target_arch = "arm") && BITMASK_STRIDE.is_multiple_of(8) {
            // SAFETY: A byte-swapped non-zero value is still non-zero.
            let swapped = unsafe { NonZeroBitMaskWord::new_unchecked(nonzero.get().swap_bytes()) };
            swapped.leading_zeros() as usize / BITMASK_STRIDE
        } else {
            nonzero.trailing_zeros() as usize / BITMASK_STRIDE
        }
    }
}

impl IntoIterator for BitMask {
    type Item = usize;
    type IntoIter = BitMaskIter;

    #[inline]
    fn into_iter(self) -> BitMaskIter {
        // A BitMask only requires each element (group of bits) to be non-zero.
        // However for iteration we need each element to only contain 1 bit.
        BitMaskIter(BitMask(self.0 & BITMASK_ITER_MASK))
    }
}

/// Iterator over the contents of a `BitMask`, returning the indices of set
/// bits.
#[derive(Copy, Clone)]
pub(crate) struct BitMaskIter(BitMask);

impl Iterator for BitMaskIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        let bit = self.0.lowest_set_bit()?;
        self.0 = self.0.remove_lowest_bit();
        Some(bit)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.0.0.count_ones() as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for BitMaskIter {
    #[inline]
    fn len(&self) -> usize {
        self.0.0.count_ones() as usize
    }
}

impl core::iter::FusedIterator for BitMaskIter {}

#[cfg(test)]
mod tests {
    use super::{BitMask, BITMASK_ITER_MASK, BITMASK_STRIDE};
    use crate::raw::Group;

    #[test]
    fn scans_first_and_last_byte() {
        let mut control = [0x17u8; Group::WIDTH];
        control[0] = 0x42;
        control[Group::WIDTH - 1] = 0x42;
        // SAFETY: `control` contains an initialized, full SIMD group.
        let group = unsafe { Group::load(control.as_ptr()) };
        let mut matches = group.match_byte(0x42).into_iter();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches.next(), Some(0));
        assert_eq!(matches.next(), Some(Group::WIDTH - 1));
        assert_eq!(matches.next(), None);
    }

    #[test]
    fn matches_each_byte_position() {
        for position in 0..Group::WIDTH {
            let mut control = [0x17u8; Group::WIDTH];
            control[position] = 0x42;
            // SAFETY: Full, initialized group.
            let group = unsafe { Group::load(control.as_ptr()) };
            let indices: Vec<_> = group.match_byte(0x42).into_iter().collect();
            assert_eq!(indices, vec![position]);
        }
    }

    #[test]
    fn sparse_and_compact_masks_iterate_in_order() {
        let raw = if BITMASK_STRIDE == 8 {
            // For NEON, one high bit is present in each 8-bit lane.
            0x8080_8080_8080_8080u64
        } else {
            u64::MAX
        };
        let active = BitMask((raw as super::BitMaskWord) & BITMASK_ITER_MASK);
        let actual: Vec<_> = active.into_iter().collect();
        let expected: Vec<_> = (0..Group::WIDTH).collect();
        assert_eq!(actual, expected);
    }
}
