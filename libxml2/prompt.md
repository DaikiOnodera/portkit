Tips for getting started:

 1. Run /init to create a CLAUDE.md file with instructions for Claude
 2. Use Claude to help with file analysis, editing, bash commands and git
 3. Be as specific as you would with another engineer for the best results
 4. ✔ Run /terminal-setup to set up terminal integration

 ※ Tip: Press Shift+Enter to send a multi-line message

> You are an expert C to Rust translator. Your task is to create:

  1. A Rust implementation (identical struct or function to the original)
  2. FFI bindings for the C version of the symbol
  3. A fuzz test that compares C and Rust implementations

  Symbol: ZopfliLZ77Store
  Kind: struct

  Project structure:
  - C source is in src/zopfli/
  - Rust project is in rust/
  - Rust source is in rust/src/
  - Rust fuzz tests are in rust/fuzz/fuzz_targets/

  Guidelines:
  - For functions: Create implementation with correct signature and implementation
  - For structs: Create identical layout with proper field types and padding
  - For typedefs: Create equivalent Rust type alias
  - Rust implementations should NOT be marked with `#[no_mangle]` or `#[export_name]`
  - Always create FFI binding for the C version
  - Use identical function and argument names as the C function
  - Assume all ifdefs are set to defined when reading C code
  - Use the same symbol names for Rust and C code

  Fuzz test requirements:
  1. Use libfuzzer_sys to generate random inputs
  2. Call both C implementation (via FFI) and Rust implementation
  3. Compare outputs and assert they are identical
  4. Handle edge cases gracefully with clear assertion messages
  5. C FFI implementations are in the zopfli::ffi module
  6. Rust implementations are in zopfli::lz77

  Create:
  1. Rust implementation stub in rust/src/lz77.rs
  2. FFI binding in rust/src/ffi.rs
  3. Fuzz test in rust/fuzz/fuzz_targets/fuzz_ZopfliLZ77Store.rs

  <fuzzing>
  ## Fuzzing Guidelines

  Only write fuzz tests for functions.
  Don't test inputs that are invalid for the C function, e.g. if an input expects a number between 0 and 30, don't test with 123123.

  When appropriate, use the C FFI bindings to initialize data, e.g. use ffi::ZopfliInitLZ77Store to initialize a ZopfliLZ77Store,
  instead of trying to manually initialize the data.

  - Use libfuzzer_sys to generate random inputs
  - Call C version first (via FFI) and then Rust version
  - Compare outputs and assert they are identical
  - Only test valid inputs for the C function.
  - Use Arbitrary to generate inputs.

  !!!! IMPORTANT !!!!
  If the C function asserts on an input, remove that input from the fuzz tests. 
  Don't try to fix the C function, or try to make the Rust function assert identically.
  !!!! IMPORTANT !!!!

  #![no_main]
  use libfuzzer_sys::fuzz_target;
  use std::os::raw::c_ushort;

  use zopfli::ffi;
  use zopfli::lz77;

  #[derive(Debug, arbitrary::Arbitrary)]
  struct FuzzInput {
      data: Vec<u8>,

      // Constrain input to a valid range, don't test on invalid inputs
      #[arbitrary(with = |u: &mut Unstructured| u.int_in_range(0..=30))]
      start: u16,

      #[arbitrary(with = |u: &mut Unstructured| u.int_in_range(0..=30))]
      end: u16,
  }

  fuzz_target!(|input: FuzzInput| {
      let size = input.data.len();
      let mut c_input = zopfli::ffi::SomeStruct { a: input.start, b: input.end };
      let mut rust_input = zopfli::ffi::SomeStruct { a: input.start, b: input.end };

      let c_result = zopfli::ffi::SomeFunction(&c_input);
      let rust_result = zopfli::ffi::SomeFunction(&rust_input);

      assert_eq!(c_result, rust_result);
  });
  </fuzzing>

  <c_declaration>
  struct ZopfliLZ77Store {
  unsigned short* litlens;
  unsigned short* dists;
  ...
  </c_declaration>

  <c_source>

  </c_source>
