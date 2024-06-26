! Copyright (c), The Regents of the University of California
! Terms of use are as specified in LICENSE.txt
program concurrent_inferences
  !! This program demonstrates how to read a neural network from a JSON file
  !! and use the network to perform concurrent inferences.
  use inference_engine_m, only : inference_engine_t, tensor_t, infer, parallel_infer
  use sourcery_m, only : string_t, command_line_t, file_t
  use assert_m, only : assert, intrinsic_array_t
  use iso_fortran_env, only : int64, real64
  use omp_lib

  implicit none

  type(string_t) network_file_name
  type(command_line_t) command_line

  network_file_name = string_t(command_line%flag_value("--network"))

  if (len(network_file_name%string())==0) then
    error stop new_line('a') // new_line('a') // &
      'Usage: fpm run --example concurrent-inferences --profile release --flag "-fopenmp" -- --network "<file-name>"'
  end if

  block 
    type(inference_engine_t) network, inference_engine
    type(tensor_t), allocatable :: inputs(:,:,:), outputs(:,:,:), outputs_elem_infer(:,:,:)
    real, allocatable :: input_components(:,:,:,:)
    real, parameter :: tolerance = 1.e-01
    integer, parameter :: lat=20, lon=350, lev=450 ! latitudes, longitudes, levels (elevations)
    integer i, j, k, dims_3(3),dims_2(2),dims_1(1)

    print *, "Constructing a new inference_engine_t object from the file " // network_file_name%string()
    inference_engine = inference_engine_t(file_t(network_file_name))

    print *,inference_engine%num_inputs()
    print *,inference_engine%num_outputs()
    print *,"Defining an array of tensor_t input objects with random normalized components"
    allocate(inputs(lon,lev,lat))
    allocate(outputs(lon,lev,lat))
    allocate(input_components(lon,lev,lat,inference_engine%num_inputs()))
    call random_number(input_components)

    do concurrent(i=1:lon, j=1:lev, k=1:lat)
      inputs(i,j,k) = tensor_t(input_components(i,j,k,:))
    end do

    block 
      integer(int64) t_start, t_finish, clock_rate

      print *,"Performing elemental inferences"
      call system_clock(t_start, clock_rate)
      outputs_elem_infer = inference_engine%infer(inputs)  ! implicit allocation of outputs array
      call system_clock(t_finish)
      print *,"Elemental inference time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)

      call assert(all(shape(outputs_elem_infer) == shape(inputs)), "all(shape(outputs) == shape(inputs))")

      print *,"Performing loop-based inference"
      call system_clock(t_start)
      do k=1,lat
        do j=1,lev
          do i=1,lon
            outputs(i,j,k) = inference_engine%infer(inputs(i,j,k))
          end do
        end do
      end do
      call system_clock(t_finish)
      print *,"Looping inference time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)

      print *,"Performing concurrent inference"
      call system_clock(t_start)
      do concurrent(i=1:lon, j=1:lev, k=1:lat)
        outputs(i,j,k) = inference_engine%infer(inputs(i,j,k))           
      end do
      call system_clock(t_finish)
      print *,"Concurrent inference time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)

      print *,"Performing concurrent inference with a non-type-bound inference procedure"
      call system_clock(t_start)
      do concurrent(i=1:lon, j=1:lev, k=1:lat)
        outputs(i,j,k) = infer(inference_engine, inputs(i,j,k))           
      end do
      call system_clock(t_finish)
      print *,"Concurrent inference time with non-type-bound procedure: ", real(t_finish - t_start, real64)/real(clock_rate, real64)

      !Copy inference_engine data to cuda version
      open(1, file = 'd_w.dat', status = 'new')
      open(2, file = 'd_b.dat', status = 'new')
      open(3, file = 'd_n.dat', status = 'new')
      open(4, file = 'd_dim.dat', status='new')
      open(5, file = 'd_range.dat', status='new')
      dims_3 = shape(inference_engine%weights_)
      write(4,*) dims_3(1),dims_3(2),dims_3(3)
      dims_2 = shape(inference_engine%biases_)
      print *,dims_2
      write(4,*) dims_2(1),dims_2(2)
      dims_1 = shape(inference_engine%nodes_)
      write(4,*) dims_1(1)
      write(4,*) inference_engine%num_inputs(), inference_engine%num_outputs()  

      !write(5,*) inference_engine%input_range_%layer_
      write(5,*) inference_engine%input_range_%minima_
      write(5,*) inference_engine%input_range_%maxima_
      !write(5,*) inference_engine%output_range_%layer_
      write(5,*) inference_engine%output_range_%minima_
      write(5,*) inference_engine%output_range_%maxima_
       

      do i=1, dims_3(1)
        do j=1, dims_3(2)
          write(1,*) inference_engine%weights_(i,j,:)
        end do
      end do

      do i=1, dims_2(1)    
        write(2,*) inference_engine%biases_(i,:)
      end do

      do i=0, dims_1(1) -1    
        write(3,*) inference_engine%nodes_(i)
      end do

      do i = 0,2
        print *,"Performing multithreading/offloading inferences"
        call system_clock(t_start, clock_rate)
        outputs = inference_engine%parallel_infer(inputs)  ! implicit allocation of outputs array
        call system_clock(t_finish)
        print *,"Multithreading/Offloading inference time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)
      end do
      do concurrent(i=1:lon, j=1:lev, k=1:lat)
        call assert(all(abs(outputs(i,j,k)%values() - outputs_elem_infer(i,j,k)%values()) < tolerance), &
          "all(openmp_offloading_outputs == outputs_elemental_infer)", &
          intrinsic_array_t(outputs(i,j,k)%values()))
      end do  
    end block
  end block

end program
