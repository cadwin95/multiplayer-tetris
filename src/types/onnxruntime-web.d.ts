declare module 'onnxruntime-web' {
  export interface InferenceSession {
    run(feeds: { [key: string]: Tensor }): Promise<{ [key: string]: Tensor }>;
  }

  export interface TensorConstructor {
    new(type: string, data: Float32Array, dims: number[]): Tensor;
  }

  export interface Tensor {
    data: Float32Array;
    dims: number[];
    type: string;
  }

  export const Tensor: TensorConstructor;
  
  export const InferenceSession: {
    create(path: string): Promise<InferenceSession>;
  };
} 