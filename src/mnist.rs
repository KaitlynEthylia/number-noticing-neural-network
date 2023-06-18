use mnist::*;

fn get(base_path: &str) {
    let Mnist {
        tst_lbl,
        tst_img,
        trn_lbl,
        trn_img,
        ..
    } = MnistBuilder::new()
        .base_path(base_path)
        .training_set_length(100)
        .test_set_length(100)
        .finalize();
}
