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
    real, allocatable :: input_components(:,:,:,:), outputs_serial(:), outputs_parallel(:)
    real, parameter :: tolerance = 1.e6
    integer :: lat, lon, lev ! latitudes, longitudes, levels (elevations)
    integer i, j, k, e, dims_w(3),dims_b(2),dims_n(1), status
    real n_operations_single_point, n_operations, error, byte_to_transfer, error_sum
    integer rep
    integer, parameter :: maxrep = 3, nPoints = 450*350*20
    real(real64) time_serial, time_parallel, time_exec, time_transf
    print *, "Constructing a new inference_engine_t object from the file " // network_file_name%string()
    inference_engine = inference_engine_t(file_t(network_file_name))
    print *, inference_engine%nodes_
    status =  copy_on_file(inference_engine)
    n_operations_single_point = 0
    do i = 1, ubound(inference_engine%nodes_,1)
      n_operations_single_point = n_operations_single_point + 2 * inference_engine%nodes_(i) * inference_engine%nodes_(i-1) + 2 * inference_engine%nodes_(i)
    end do
    dims_w = shape(inference_engine%weights_)
    dims_b = shape(inference_engine%biases_)
    dims_n = shape(inference_engine%nodes_)
    allocate(outputs_serial(inference_engine%num_outputs()))
    allocate(outputs_parallel(inference_engine%num_outputs()))
    open(unit=1, file="results.csv", status='replace', action='write')
    write(1, '(A)') 'NPoints;TimeSer;TimeParal;TimeExec;GFLOPSSer;GFLOPSExec;SpeedUpParal;SpeedupExec;RelError;Bandwidth'

    print *,"Defining an array of tensor_t input objects with random normalized components"
    lev = 1
    lat = 1
    do lon=1,nPoints+1,50000 
      n_operations = (lat * lev * lon) * n_operations_single_point / 10**9
      byte_to_transfer = ((lat*lon*lev)*(inference_engine%num_inputs()+inference_engine%num_outputs()) + dims_n(1)&
                  + dims_b(1)*dims_b(2) + dims_w(1)*dims_w(2)+dims_w(3) + 2*inference_engine%num_inputs() + 2*inference_engine%num_outputs())/10**9
      allocate(inputs(lon,lev,lat))
      allocate(outputs(lon,lev,lat))
      allocate(input_components(lon,lev,lat,inference_engine%num_inputs()))
      call random_number(input_components)

      do concurrent(i=1:lon, j=1:lev, k=1:lat)
        inputs(i,j,k) = tensor_t(input_components(i,j,k,:))
      end do

      block 
        integer(int64) t_start, t_finish, clock_rate
        real(real64) t_exec, t_transf
        time_serial = 0
        time_parallel = 0
        time_exec = 0
        error_sum = 0
        time_transf = 0
        print *, lon
        do rep=1,maxrep
          error = 0
          !print *,"Performing elemental inferences"
          call system_clock(t_start, clock_rate)
          outputs_elem_infer = inference_engine%infer(inputs)  ! implicit allocation of outputs array
          call system_clock(t_finish)
          time_serial = time_serial + real(t_finish - t_start, real64)/real(clock_rate, real64)
          !print *,"Elemental inference time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)
          !call assert(all(shape(outputs_elem_infer) == shape(inputs)), "all(shape(outputs) == shape(inputs))")

          ! print *,"Performing loop-based inference"
          ! call system_clock(t_start)
          ! do k=1,lat
          !   do j=1,lev
          !     do i=1,lon
          !       outputs(i,j,k) = inference_engine%infer(inputs(i,j,k))
          !     end do
          !   end do
          ! end do
          ! call system_clock(t_finish)
          ! print *,"Looping inference time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)

          ! print *,"Performing concurrent inference"
          ! call system_clock(t_start)
          ! do concurrent(i=1:lon, j=1:lev, k=1:lat)
          !   outputs(i,j,k) = inference_engine%infer(inputs(i,j,k))           
          ! end do
          ! call system_clock(t_finish)
          ! print *,"Concurrent inference time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)

          ! print *,"Performing concurrent inference with a non-type-bound inference procedure"
          ! call system_clock(t_start)
          ! do concurrent(i=1:lon, j=1:lev, k=1:lat)
          !   outputs(i,j,k) = infer(inference_engine, inputs(i,j,k))           
          ! end do
          ! call system_clock(t_finish)
          ! print *,"Concurrent inference time with non-type-bound procedure: ", real(t_finish - t_start, real64)/real(clock_rate, real64)
          
          !print *,"Performing multithreading/offloading inferences"
          call system_clock(t_start, clock_rate)
          call inference_engine%parallel_infer(inputs,outputs,t_exec)  ! implicit allocation of outputs array
          call system_clock(t_finish)
          time_exec = time_exec + t_exec
          !time_transf = time_transf + t_transf
          time_parallel = time_parallel + real(t_finish - t_start, real64)/real(clock_rate, real64)
          !print *,"Multithreading/Offloading inference time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)
          !print *, t_exec
          do i=1,lon
            do j=1,lev
              do k=1,lat
                outputs_serial = outputs_elem_infer(i,j,k)%values()
                outputs_parallel = outputs(i,j,k)%values()
                do e=1,inference_engine%num_outputs()
                  error = error +(abs(outputs_parallel(e)- outputs_serial(e))*(tolerance))/abs(outputs_serial(e));
                end do
              end do
            end do
          end do  
          error_sum = error_sum + error/(lon*lev*lat*inference_engine%num_outputs())
        end do  
      end block
      deallocate(inputs)
      deallocate(outputs)
      deallocate(input_components)
      time_parallel = time_parallel / maxrep
      time_serial = time_serial / maxrep
      time_exec = time_exec / maxrep
      error_sum = error_sum / maxrep
      write(1, '(I7,";",F12.8,";",F12.8,";",F12.8,";",F12.8,";",F12.8,";",F12.8,";",F12.8,";",F12.8,";",F12.8,";",F12.8)') &
      lon, time_serial, time_parallel, time_exec, real(n_operations)/time_serial, n_operations/time_exec, time_serial/time_parallel, time_serial/time_exec, error_sum, byte_to_transfer/(time_parallel-time_exec) 
    end do
    close(1)
  end block

  contains
  function copy_on_file(inference_engine) result(status)
    implicit none
    type(inference_engine_t), intent(in) :: inference_engine
    integer :: status
    integer i,j,k,dims_3(3),dims_2(2),dims_1(1)
    !Copy inference_engine data to cuda version
    open(1, file = 'd_w.dat', status = 'replace')
    open(2, file = 'd_b.dat', status = 'replace')
    open(3, file = 'd_n.dat', status = 'replace')
    open(4, file = 'd_dim.dat', status='replace')
    open(5, file = 'd_range.dat', status='replace')
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

    status = 1

    do i=1,5
      close(i)
    end do
    return 
  end function copy_on_file

end program
