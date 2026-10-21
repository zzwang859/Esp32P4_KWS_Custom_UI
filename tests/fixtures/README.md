`esp32p4_reference.npz` contains independent golden outputs from the provided
`E:/esp32-p4-onnx/model_toolchain/ds_tc_resnet` ONNX models: 180 sequential
backend frames and 100 audio/MFCC frames. The source full model SHA-256 is
`7ff161caded03c8beef17d3da0933bbe7143f731f5cc76421cdc25664a9f1a56`.

`weight__*` arrays map the reference's folded convolution/bias weights into the
training module, with BatchNorm initialized as the corresponding affine
identity. These are test-only reference weights; normal training initializes
its own weights. `expected_logits` and `expected_mfcc` were computed by ONNX
Runtime from the reference backend and extracted audio frontend, not from the
implementation under test. The fixture covers startup silence and crossing
the 98-frame pooling window boundary.
