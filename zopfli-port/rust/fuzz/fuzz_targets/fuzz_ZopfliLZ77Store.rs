#![no_main]
use libfuzzer_sys::fuzz_target;
use zopfli::ffi;
use zopfli::lz77;
use std::ptr;
use std::mem;
use arbitrary::Unstructured;

#[derive(Debug, arbitrary::Arbitrary)]
struct FuzzInput {
    data: Vec<u8>,
    operations: Vec<StoreOperation>,
}

#[derive(Debug, arbitrary::Arbitrary)]
enum StoreOperation {
    InitStore,
    StoreLitLenDist {
        #[arbitrary(with = |u: &mut Unstructured| u.int_in_range(1..=258))]
        length: u16,
        #[arbitrary(with = |u: &mut Unstructured| u.int_in_range(0..=32768))] 
        dist: u16,
        pos_offset: u8, // Will be used as pos % data.len()
    },
    GetByteRange {
        start_offset: u8,
        end_offset: u8,
    },
    GetHistogram {
        start_offset: u8,
        end_offset: u8,
    },
    CopyStore,
    AppendStore,
}

unsafe fn init_empty_store(store: *mut ffi::ZopfliLZ77Store, data: &[u8]) {
    ffi::ZopfliInitLZ77Store(data.as_ptr(), store);
}

unsafe fn compare_stores_basic(store1: *const ffi::ZopfliLZ77Store, store2: *const ffi::ZopfliLZ77Store) {
    let store1 = &*store1;
    let store2 = &*store2;
    
    assert_eq!(store1.size, store2.size, "Store sizes differ");
    assert_eq!(store1.data, store2.data, "Data pointers differ");
    
    if store1.size > 0 {
        // Compare the main arrays
        let litlens1 = std::slice::from_raw_parts(store1.litlens, store1.size as usize);
        let litlens2 = std::slice::from_raw_parts(store2.litlens, store2.size as usize);
        assert_eq!(litlens1, litlens2, "litlens arrays differ");
        
        let dists1 = std::slice::from_raw_parts(store1.dists, store1.size as usize);
        let dists2 = std::slice::from_raw_parts(store2.dists, store2.size as usize);
        assert_eq!(dists1, dists2, "dists arrays differ");
        
        let pos1 = std::slice::from_raw_parts(store1.pos, store1.size as usize);
        let pos2 = std::slice::from_raw_parts(store2.pos, store2.size as usize);
        assert_eq!(pos1, pos2, "pos arrays differ");
        
        let ll_symbol1 = std::slice::from_raw_parts(store1.ll_symbol, store1.size as usize);
        let ll_symbol2 = std::slice::from_raw_parts(store2.ll_symbol, store2.size as usize);
        assert_eq!(ll_symbol1, ll_symbol2, "ll_symbol arrays differ");
        
        let d_symbol1 = std::slice::from_raw_parts(store1.d_symbol, store1.size as usize);
        let d_symbol2 = std::slice::from_raw_parts(store2.d_symbol, store2.size as usize);
        assert_eq!(d_symbol1, d_symbol2, "d_symbol arrays differ");
        
        // Compare histograms
        let llsize = 288 * lz77::CeilDiv(store1.size as usize, 288);
        let dsize = 32 * lz77::CeilDiv(store1.size as usize, 32);
        
        if !store1.ll_counts.is_null() && !store2.ll_counts.is_null() {
            let ll_counts1 = std::slice::from_raw_parts(store1.ll_counts, llsize);
            let ll_counts2 = std::slice::from_raw_parts(store2.ll_counts, llsize);
            assert_eq!(ll_counts1, ll_counts2, "ll_counts arrays differ");
        }
        
        if !store1.d_counts.is_null() && !store2.d_counts.is_null() {
            let d_counts1 = std::slice::from_raw_parts(store1.d_counts, dsize);
            let d_counts2 = std::slice::from_raw_parts(store2.d_counts, dsize);
            assert_eq!(d_counts1, d_counts2, "d_counts arrays differ");
        }
    }
}

fuzz_target!(|input: FuzzInput| {
    if input.data.is_empty() || input.operations.is_empty() {
        return;
    }
    
    unsafe {
        let mut c_store: ffi::ZopfliLZ77Store = mem::zeroed();
        let mut rust_store: ffi::ZopfliLZ77Store = mem::zeroed();
        
        // Initialize both stores
        init_empty_store(&mut c_store, &input.data);
        init_empty_store(&mut rust_store, &input.data);
        
        for op in &input.operations {
            match op {
                StoreOperation::InitStore => {
                    // Re-initialize (should be safe to call multiple times)
                    ffi::ZopfliCleanLZ77Store(&mut c_store);
                    ffi::ZopfliCleanLZ77Store(&mut rust_store);
                    init_empty_store(&mut c_store, &input.data);
                    init_empty_store(&mut rust_store, &input.data);
                }
                
                StoreOperation::StoreLitLenDist { length, dist, pos_offset } => {
                    if input.data.is_empty() {
                        continue;
                    }
                    let pos = (*pos_offset as usize) % input.data.len();
                    
                    // Only store if the position is valid and the distance/length make sense
                    if *dist == 0 {
                        // Literal: length should be the literal value (0-255)
                        if *length <= 255 {
                            ffi::ZopfliStoreLitLenDist(*length, *dist, pos, &mut c_store);
                            lz77::ZopfliStoreLitLenDist(*length, *dist, pos, &mut rust_store);
                        }
                    } else if *dist > 0 && pos >= *dist as usize {
                        // Match: ensure we don't go out of bounds
                        let actual_length = (*length as usize).min(input.data.len() - pos);
                        let actual_length = actual_length.min(258); // Max match length
                        if actual_length >= 3 { // Min match length
                            ffi::ZopfliStoreLitLenDist(actual_length as u16, *dist, pos, &mut c_store);
                            lz77::ZopfliStoreLitLenDist(actual_length as u16, *dist, pos, &mut rust_store);
                        }
                    }
                }
                
                StoreOperation::GetByteRange { start_offset, end_offset } => {
                    if c_store.size == 0 {
                        continue;
                    }
                    let start = (*start_offset as usize) % (c_store.size as usize + 1);
                    let end = start + ((*end_offset as usize) % (c_store.size as usize - start + 1));
                    
                    let c_result = ffi::ZopfliLZ77GetByteRange(&c_store, start, end);
                    let rust_result = lz77::ZopfliLZ77GetByteRange(&rust_store, start, end);
                    
                    assert_eq!(c_result, rust_result, "GetByteRange results differ for range {}..{}", start, end);
                }
                
                StoreOperation::GetHistogram { start_offset, end_offset } => {
                    if c_store.size == 0 {
                        continue;
                    }
                    let start = (*start_offset as usize) % (c_store.size as usize + 1);
                    let end = start + ((*end_offset as usize) % (c_store.size as usize - start + 1));
                    
                    let mut c_ll_counts = [0usize; 288];
                    let mut c_d_counts = [0usize; 32];
                    let mut rust_ll_counts = [0usize; 288];
                    let mut rust_d_counts = [0usize; 32];
                    
                    ffi::ZopfliLZ77GetHistogram(&c_store, start, end, c_ll_counts.as_mut_ptr(), c_d_counts.as_mut_ptr());
                    lz77::ZopfliLZ77GetHistogram(&rust_store, start, end, rust_ll_counts.as_mut_ptr(), rust_d_counts.as_mut_ptr());
                    
                    assert_eq!(c_ll_counts, rust_ll_counts, "LL histogram differs for range {}..{}", start, end);
                    assert_eq!(c_d_counts, rust_d_counts, "D histogram differs for range {}..{}", start, end);
                }
                
                StoreOperation::CopyStore => {
                    let mut c_dest_store: ffi::ZopfliLZ77Store = mem::zeroed();
                    let mut rust_dest_store: ffi::ZopfliLZ77Store = mem::zeroed();
                    
                    ffi::ZopfliCopyLZ77Store(&c_store, &mut c_dest_store);
                    lz77::ZopfliCopyLZ77Store(&rust_store, &mut rust_dest_store);
                    
                    compare_stores_basic(&c_dest_store, &rust_dest_store);
                    
                    ffi::ZopfliCleanLZ77Store(&mut c_dest_store);
                    ffi::ZopfliCleanLZ77Store(&mut rust_dest_store);
                }
                
                StoreOperation::AppendStore => {
                    // Create a source store to append from
                    let mut c_source_store: ffi::ZopfliLZ77Store = mem::zeroed();
                    let mut rust_source_store: ffi::ZopfliLZ77Store = mem::zeroed();
                    init_empty_store(&mut c_source_store, &input.data);
                    init_empty_store(&mut rust_source_store, &input.data);
                    
                    // Add a simple literal to the source store
                    if !input.data.is_empty() {
                        let literal = input.data[0] as u16;
                        ffi::ZopfliStoreLitLenDist(literal, 0, 0, &mut c_source_store);
                        lz77::ZopfliStoreLitLenDist(literal, 0, 0, &mut rust_source_store);
                    }
                    
                    // Create target stores (copy of current stores)
                    let mut c_target_store: ffi::ZopfliLZ77Store = mem::zeroed();
                    let mut rust_target_store: ffi::ZopfliLZ77Store = mem::zeroed();
                    ffi::ZopfliCopyLZ77Store(&c_store, &mut c_target_store);
                    lz77::ZopfliCopyLZ77Store(&rust_store, &mut rust_target_store);
                    
                    // Append the source to the target
                    ffi::ZopfliAppendLZ77Store(&c_source_store, &mut c_target_store);
                    lz77::ZopfliAppendLZ77Store(&rust_source_store, &mut rust_target_store);
                    
                    compare_stores_basic(&c_target_store, &rust_target_store);
                    
                    ffi::ZopfliCleanLZ77Store(&mut c_source_store);
                    ffi::ZopfliCleanLZ77Store(&mut rust_source_store);
                    ffi::ZopfliCleanLZ77Store(&mut c_target_store);
                    ffi::ZopfliCleanLZ77Store(&mut rust_target_store);
                }
            }
            
            // After each operation, verify the stores are still equivalent
            compare_stores_basic(&c_store, &rust_store);
        }
        
        // Clean up
        ffi::ZopfliCleanLZ77Store(&mut c_store);
        ffi::ZopfliCleanLZ77Store(&mut rust_store);
    }
});