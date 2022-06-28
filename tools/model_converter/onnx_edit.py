#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to modify converted YOLO ONNX model to remove "Transpose" OP
at head. This can be used when you apply "--inputs_as_nchw" in keras to onnx
convert, so that you can get a full "NCHW" layout ONNX model for better PyTorch
style compatibility
"""
import os, sys, argparse
import onnx


def onnx_edit(input_model, output_model):
    onnx_model = onnx.load(input_model)
    graph = onnx_model.graph
    node  = graph.node

    for out in graph.output:
        print('Editing output tensor:', out.name)
        update_flag = False
        for n in node:
            if n.op_type == 'Transpose' and len(n.output) == 1 and out.name == n.output[0]:
                assert len(n.input) == 1, 'invalid input number for Transpose.'

                # Option 1: find OP before the "Transpose", and update its output
                # tensor name to the graph output
                for n_prev in node:
                    if len(n_prev.output) == 1 and n_prev.output[0] == n.input[0]:
                        print('Found the node before Transpose: {}, will use its output as final output'.format(n_prev.name))
                        n_prev.output[0] = out.name

                # Option 2: change graph output to the OP output before "Transpose"
                #out.name = n.input[0]

                # update output shape from NHWC to NCHW, since we delete a "Transpose"
                output_shape = out.type.tensor_type.shape.dim

                # switch from NHWC to NCHW
                output_height = output_shape[1].dim_value
                output_width = output_shape[2].dim_value
                output_channel = output_shape[3].dim_value

                output_shape[1].dim_value = output_channel
                output_shape[2].dim_value = output_height
                output_shape[3].dim_value = output_width

                # remove Transpose OP node
                graph.node.remove(n)
                update_flag = True

        if not update_flag:
            print('Fail to update:', out.name)


    # save changed model
    #graph = onnx.helper.make_graph(graph.node, graph.name, graph.input, graph.output, graph.initializer)
    #info_model = onnx.helper.make_model(graph)
    #onnx_model = onnx.shape_inference.infer_shapes(info_model)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, output_model)
    print('Done.')



def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='edit YOLO onnx model to delete the Transpose OP at head')
    parser.add_argument('--input_model', help='input YOLO onnx model file to edit', type=str, required=True)
    parser.add_argument('--output_model', help='output onnx model file to save', type=str, required=True)

    args = parser.parse_args()

    onnx_edit(args.input_model, args.output_model)


if __name__ == "__main__":
    main()
