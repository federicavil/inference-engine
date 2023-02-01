! Copyright (c), The Regents of the University of California
! Terms of use are as specified in LICENSE.txt
program convert_to_json
  !! This program demonstrates how to read a neural network in a custom file format
  !! into an inference_engine_t object, convert it to a JSON format, and write the
  !! resulting JSON objects to a file.
  use assert_m, only : assert
  use command_line_m, only : command_line_t
  use inference_engine_m, only : inference_engine_t
  use string_m, only : string_t
  use sigmoid_m, only : sigmoid_t
  use matmul_m, only : matmul_t
  use kind_parameters_m, only : rkind
  use file_m, only : file_t
  implicit none

  type(inference_engine_t) inference_engine
  type(command_line_t) command_line
  character(len=:), allocatable :: input_file_name
  type(file_t) json_output_file

  input_file_name =  command_line%flag_value("--input-file")

  if (len(input_file_name)==0) then
    error stop new_line('a') // new_line('a') // &
      'Usage: ./build/run-fpm.sh run --example convert-to-json -- --input-file "<file-name>"' 
  end if

  print *,"Defining an inference_engine_t object by reading the file '"//input_file_name//"'"
  call inference_engine%read_network(string_t(input_file_name), sigmoid_t(), matmul_t())

  print *,"num_outputs = ", inference_engine%num_outputs()
  print *,"num_hidden_layers = ", inference_engine%num_hidden_layers()
  print *,"neurons_per_layer = ", inference_engine%neurons_per_layer()

  associate(num_inputs => inference_engine%num_inputs())
    print *,"num_inputs = ", num_inputs
    block
      real(rkind), parameter :: inputs(*) = &
        [real(rkind):: 0.0079, 1.4429e-12, 0.0000e+00, 0.0000e+00, 0.0000e+00, 4.4941e-04, 0.0000e+00, 0., 282.2671, 71541.9766]

      call assert(num_inputs==size(inputs),"main: num_inputs==size(inputs)", num_inputs)
      print *, inference_engine%infer(inputs)
    end block
  end associate

  print *, "Converting an inference_engine_t object to a file_t object."
  json_output_file = inference_engine%to_json()

  associate(output_file_name => string_t(input_file_name // ".json"))
    print *, "Writing an inference_engine_t object to the file '"//output_file_name%string()//"'."
    call json_output_file%write_lines(output_file_name)
  end associate

end program
