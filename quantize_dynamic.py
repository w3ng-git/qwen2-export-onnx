from onnxruntime.quantization import QuantType, quantize_dynamic

print('======onnx quantization begin======')

quantize_dynamic(
        model_input='./output_onnx/language-model.onnx',
        model_output='./output_onnx/a.onnx',
        # op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
)
