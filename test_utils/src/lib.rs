#[macro_export]
/// Convenience macro for testing single functions
macro_rules! compile_and_run {
    ($compiler:ident, $func:ident, $state:ident) => {
        unsafe {
            let fptr = $func.compile().unwrap().as_raw();
            $compiler
                .compile_entry_point()
                .unwrap()
                .call(&mut $state, fptr);
        }
    };
}
