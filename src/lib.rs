mod context;
pub mod tensor;

#[macro_export]
macro_rules! assert_rel_eq_arr {
    ($actual:expr, $expected:expr) => {
        assert_eq!($actual.shape(), $expected.shape());
        ndarray::Zip::from(&$actual)
            .and(&$expected)
            .for_each(|v, w| {
                assert_relative_eq!(v, w);
            });
    };
}
