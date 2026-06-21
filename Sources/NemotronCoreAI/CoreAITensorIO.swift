import CoreAI
import Foundation

enum CoreAITensorIO {
    static func inputDescriptor(_ function: InferenceFunction, named name: String) throws -> NDArrayDescriptor {
        guard case .ndArray(let descriptor) = function.descriptor.inputDescriptor(of: name) else {
            throw NemotronError.inference(
                "missing NDArray input descriptor for '\(name)' in \(function.descriptor.name)")
        }
        return descriptor
    }

    static func floatInput(
        _ values: [Float],
        shape: [Int],
        function: InferenceFunction,
        name: String
    ) throws -> NDArray {
        let base = try inputDescriptor(function, named: name)
        let descriptor = base.resolvingDynamicDimensions(shape)
        guard descriptor.shape == shape else {
            throw NemotronError.inference(
                "input '\(name)' for \(function.descriptor.name) resolved as \(descriptor.shape), expected \(shape)"
            )
        }
        guard values.count == shape.reduce(1, *) else {
            throw NemotronError.inference("input '\(name)' value count does not match \(shape)")
        }
        var array = NDArray(descriptor: descriptor)
        switch descriptor.scalarType {
        case .float16:
            var view = array.mutableView(as: Float16.self)
            view.copyElements(fromContentsOf: values.lazy.map(Float16.init))
        case .float32:
            var view = array.mutableView(as: Float.self)
            view.copyElements(fromContentsOf: values)
        default:
            throw NemotronError.unsupportedScalarType("\(descriptor.scalarType) for '\(name)'")
        }
        return array
    }

    static func zeroFloatInput(
        shape: [Int],
        function: InferenceFunction,
        name: String
    ) throws -> NDArray {
        try floatInput(
            [Float](repeating: 0, count: shape.reduce(1, *)),
            shape: shape,
            function: function,
            name: name
        )
    }

    static func int32Input(
        _ values: [Int32],
        shape: [Int],
        function: InferenceFunction,
        name: String
    ) throws -> NDArray {
        let base = try inputDescriptor(function, named: name)
        let descriptor = base.resolvingDynamicDimensions(shape)
        guard descriptor.shape == shape, descriptor.scalarType == .int32 else {
            throw NemotronError.inference(
                "input '\(name)' for \(function.descriptor.name) must be int32 \(shape); found \(descriptor.scalarType) \(descriptor.shape)"
            )
        }
        guard values.count == shape.reduce(1, *) else {
            throw NemotronError.inference("input '\(name)' value count does not match \(shape)")
        }
        var array = NDArray(descriptor: descriptor)
        var view = array.mutableView(as: Int32.self)
        view.copyElements(fromContentsOf: values)
        return array
    }

    static func take(
        _ outputs: inout InferenceFunction.Outputs,
        named name: String,
        function: String
    ) throws -> NDArray {
        guard let value = outputs.remove(name), let array = value.ndArray else {
            throw NemotronError.inference("function '\(function)' did not return NDArray output '\(name)'")
        }
        return array
    }

    static func flattenFloat(_ array: NDArray) throws -> [Float] {
        switch array.scalarType {
        case .float16:
            return flatten(array, as: Float16.self)
        case .float32:
            return flatten(array, as: Float.self)
        default:
            throw NemotronError.unsupportedScalarType("\(array.scalarType)")
        }
    }

    static func firstInteger(_ array: NDArray) throws -> Int {
        guard array.shape.reduce(1, *) > 0 else {
            throw NemotronError.inference("integer output is empty")
        }
        switch array.scalarType {
        case .int32:
            return Int(readFirst(array, as: Int32.self))
        case .int64:
            return Int(readFirst(array, as: Int64.self))
        default:
            throw NemotronError.unsupportedScalarType("\(array.scalarType) for integer output")
        }
    }

    static func argmax(_ array: NDArray) throws -> Int {
        let values = try flattenFloat(array)
        guard let first = values.first else {
            throw NemotronError.inference("cannot take argmax of empty logits")
        }
        var bestIndex = 0
        var bestValue = first
        for index in values.indices.dropFirst() where values[index] > bestValue {
            bestIndex = index
            bestValue = values[index]
        }
        return bestIndex
    }

    private static func flatten<T: BinaryFloatingPoint & BitwiseCopyable>(
        _ array: NDArray,
        as type: T.Type
    ) -> [Float] {
        let outerShape = array.shape
        let total = outerShape.reduce(1, *)
        let rank = outerShape.count
        var result = [Float](repeating: 0, count: total)
        array.view(as: type).withUnsafePointer { pointer, shape, strides in
            var expectedStride = 1
            var isContiguous = true
            for dimension in (0..<rank).reversed() {
                if strides[dimension] != expectedStride {
                    isContiguous = false
                    break
                }
                expectedStride *= shape[dimension]
            }
            if isContiguous {
                for index in 0..<total {
                    result[index] = Float(pointer[index])
                }
                return
            }

            var indices = [Int](repeating: 0, count: rank)
            for resultIndex in 0..<total {
                var sourceOffset = 0
                for dimension in 0..<rank {
                    sourceOffset += indices[dimension] * strides[dimension]
                }
                result[resultIndex] = Float(pointer[sourceOffset])
                var dimension = rank - 1
                while dimension >= 0 {
                    indices[dimension] += 1
                    if indices[dimension] < shape[dimension] { break }
                    indices[dimension] = 0
                    dimension -= 1
                }
            }
        }
        return result
    }

    private static func readFirst<T: BitwiseCopyable>(_ array: NDArray, as type: T.Type) -> T {
        array.view(as: type).withUnsafePointer { pointer, _, _ in pointer[0] }
    }
}
