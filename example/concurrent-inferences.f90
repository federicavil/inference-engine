! Copyright (c), The Regents of the University of California
! Terms of use are as specified in LICENSE.txt
program concurrent_inferences
  !! This program demonstrates how to read a neural network from a JSON file
  !! and use the network to perform concurrent inferences.
  use inference_engine_m, only : inference_engine_t, tensor_t, infer
  use activation_strategy_m, only : activation_strategy_t
  use sourcery_m, only : string_t, command_line_t, file_t
  use assert_m, only : assert, intrinsic_array_t
  use iso_fortran_env, only : int64, real64
  use kind_parameters_m, only : rkind
  use, intrinsic :: omp_lib
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
    type(tensor_t), allocatable :: inputs(:,:,:), outputs_elem_infer(:,:,:), outputs(:,:,:)
    class(activation_strategy_t), allocatable :: act_strategy
    real, allocatable :: input_components(:,:,:,:), output_components(:,:,:,:)
    integer, parameter :: lat=350, lon=450, lev=20 ! latitudes, longitudes, levels (elevations)
    integer i, j, k, ij,jk,l
    real, parameter :: tolerance = 1.e-06
    real, parameter :: tolerance_2 = 1.e-02
    integer, allocatable :: n(:)
    real(rkind), allocatable :: w(:,:,:), b(:,:)
    real(rkind), allocatable :: a(:,:)
    integer :: output_level
    print *, "Constructing a new inference_engine_t object from the file " // network_file_name%string()
    inference_engine = inference_engine_t(file_t(network_file_name))
    !inference_engine_dev = inference_engine_t(file_t(network_file_name))

    n = inference_engine%nodes_
    w = inference_engine%weights_
    b = inference_engine%biases_
    output_level = ubound(n,1)
    act_strategy = inference_engine%activation_strategy_
    allocate(a(maxval(n),0:ubound(n,1)))
    print *,"Defining an array of tensor_t input objects with random normalized components"
    allocate(inputs(lat,lon,lev))
    allocate(input_components(lat,lon,lev,inference_engine%num_inputs()))
    allocate(output_components(lat,lon,lev,inference_engine%num_outputs()))
    call random_number(input_components)
    !allocate(inputs_dev(lat,lon,lev))
    !allocate(outputs_dev(lat,lon,lev))
    do concurrent(i=1:lat, j=1:lon, k=1:lev)
      inputs(i,j,k) = tensor_t(input_components(i,j,k,:))
      !inputs_dev(i,j,k) = tensor_t(input_components(i,j,k,:))
    end do
    
    !inputs_dev = inputs

    block 
      integer(int64) t_start, t_finish, clock_rate

      print *,"Performing elemental inferences"
      call system_clock(t_start, clock_rate)
      associate(outputs_tensors => inference_engine%infer(inputs))
        ! added this one
        outputs_elem_infer = outputs_tensors
      end associate
      call system_clock(t_finish)
      print *,"Elemental inference time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)
      call assert(all(shape(outputs_elem_infer) == shape(inputs)), "all(shape(outputs) == shape(inputs))")
      allocate(outputs(lat,lon,lev))
      
      print *,"Performing loop-based inference"
      call system_clock(t_start)
      do k=1,lev
        do j=1,lon
          do i=1,lat
            outputs(i,j,k) = inference_engine%infer(inputs(i,j,k))
            !outputs_dev(i,j,k) = outputs(i,j,k)
          end do
        end do
      end do
      call system_clock(t_finish)
      print *,"Looping inference time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)

      !Looping inference test
      do concurrent(i=1:lat, j=1:lon, k=1:lev)
        call assert(all(abs(outputs(i,j,k)%values() - outputs_elem_infer(i,j,k)%values()) < tolerance), &
          "all(looping_outputs == outputs_elemental_infer)")
      end do  

      print *,"Performing concurrent inference"
      call system_clock(t_start)
      do concurrent(i=1:lat, j=1:lon, k=1:lev)
        outputs(i,j,k) = inference_engine%infer(inputs(i,j,k))           
      end do
      call system_clock(t_finish)
      print *,"Concurrent inference time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)

      !Concurrent inference test
      do concurrent(i=1:lat, j=1:lon, k=1:lev)
        call assert(all(abs(outputs(i,j,k)%values() - outputs_elem_infer(i,j,k)%values()) < tolerance), &
          "all(concurrent_outputs == outputs_elemental_infer)")
      end do  

      print *,"Performing concurrent inference with a non-type-bound inference procedure"
      call system_clock(t_start)
      do concurrent(i=1:lat, j=1:lon, k=1:lev)
        outputs(i,j,k) = infer(inference_engine, inputs(i,j,k))           
      end do
      call system_clock(t_finish)
      print *,"Concurrent inference time with non-type-bound procedure: ", real(t_finish - t_start, real64)/real(clock_rate, real64)

      !Concurrent inference with non-type-bound procedure test
      do concurrent(i=1:lat, j=1:lon, k=1:lev)
        call assert(all(abs(outputs(i,j,k)%values() - outputs_elem_infer(i,j,k)%values()) < tolerance), &
          "all(concurrent_with_non_type_bound_proc_outputs == outputs_elemental_infer)")
      end do  
      
      print *, "performing inference with openmp multi-threading"
      call system_clock(t_start)
      !$omp parallel do default(none) shared(inputs, outputs, inference_engine) schedule(static,1)
      do j=1,lon
        do k=1,lev
          do i=1,lat
            outputs(i,j,k) = inference_engine%infer(inputs(i,j,k))
          end do
        end do
      end do
      !$omp end parallel do
      call system_clock(t_finish)
      print *,"Concurrent inference time with openmp multi-threading: ", real(t_finish - t_start, real64)/real(clock_rate)

      !Openmp multithreading inference test
      do concurrent(i=1:lat, j=1:lon, k=1:lev)
        call assert(all(abs(outputs(i,j,k)%values() - outputs_elem_infer(i,j,k)%values()) < tolerance), &
          "all(openmp_outputs == outputs_elemental_infer)")
      end do

      print *, "performing inference with omp multithreading and inline code"
      call system_clock(t_start)
      !$omp parallel do default(none) shared(input_components, output_components, n,w,b,output_level) private(a)
        do j=1,lon
          do k=1,lev
            do i=1,lat
              a(1:n(0),0) = input_components(i,j,k,:)
              feed_forward: &
              do l = 1, output_level
                a(1:n(l),l) = 1./(1.+exp(-(matmul(w(1:n(l),1:n(l-1),l), a(1:n(l-1),l-1)) + b(1:n(l),l))))
              end do feed_forward
              output_components(i,j,k,:) = a(1:n(output_level), output_level)
            end do
          end do
        end do
      !$omp end parallel do
      call system_clock(t_finish)
      print *,"Concurrent inference time with omp multithreading and inline code: ", & 
      real(t_finish - t_start, real64)/real(clock_rate)
      
      do concurrent(i=1:lat, j=1:lon, k=1:lev)
        outputs(i,j,k) = tensor_t(output_components(i,j,k,:))
      end do

      !Openmp offloading inference test
      do concurrent(i=1:lat, j=1:lon, k=1:lev)
        call assert(all(abs(outputs(i,j,k)%values() - outputs_elem_infer(i,j,k)%values()) < tolerance), &
          "all(openmp_offloading_outputs == outputs_elemental_infer)", &
          intrinsic_array_t(outputs(i,j,k)%values()))
      end do  
      
      print *, "performing inference with openmp offloading"
      call system_clock(t_start)
      !$omp target map(from:output_components) map(to:input_components,n,w,b,a)
      !$omp teams distribute parallel do shared (input_components, output_components, n, w, b, output_level) firstprivate(a) &
      !$omp collapse(3)
        do j=1,lon
          do k=1,lev
            do i=1,lat
              a(1:n(0),0) = input_components(i,j,k,:)
              do l = 1, output_level
                a(1:n(l),l) = 1./(1.+exp(-(matmul(w(1:n(l),1:n(l-1),l), a(1:n(l-1),l-1)) + b(1:n(l),l))))
              end do 
              output_components(i,j,k,:) = a(1:n(output_level), output_level)
            end do
          end do
        end do
      !$omp end teams distribute parallel do
      !$omp end target
      call system_clock(t_finish)
      print *,"Concurrent inference time with openmp offloading: ", real(t_finish - t_start, real64)/real(clock_rate)
      
      do concurrent(i=1:lat, j=1:lon, k=1:lev)
        outputs(i,j,k) = tensor_t(output_components(i,j,k,:))
      end do

      !Openmp offloading inference test
      do concurrent(i=1:lat, j=1:lon, k=1:lev)
        call assert(all(abs(outputs(i,j,k)%values() - outputs_elem_infer(i,j,k)%values()) < tolerance), &
          "all(openmp_offloading_outputs == outputs_elemental_infer)", &
          intrinsic_array_t(outputs(i,j,k)%values()))
      end do  

      print *,"Performing inference with openacc"
      call system_clock(t_start)
      !$acc data copyout(output_components) copyin(input_components,n,w,b) create(a)
      !$acc parallel loop collapse(3)
      !!$acc parallel loop gang num_gangs(lat*lon) num_workers(lev) 
      
      do i=1,lat
      ! do ij=1,lat*lon
      !   j = (ij/lat) +1
      !   i = mod(ij,lat)
        !!$acc loop worker
        do j=1,lon
          !!$acc loop worker
          do k=1,lev
            a(1:n(0),0) = input_components(i,j,k,:)
            feedforward: &
            do l = 1, output_level
              a(1:n(l),l) = 1./(1.+exp(-(matmul(w(1:n(l),1:n(l-1),l), a(1:n(l-1),l-1)) + b(1:n(l),l))))
            end do feedforward
            output_components(i,j,k,:) = a(1:n(output_level), output_level)
          end do
        end do
      end do
      !!$acc end parallel
      !$acc end data
      call system_clock(t_finish)
      print *,"Openacc offloading time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)

      do concurrent(i=1:lat, j=1:lon, k=1:lev)
        outputs(i,j,k) = tensor_t(output_components(i,j,k,:))
      end do

      !Openmp offloading inference test
      do concurrent(i=1:lat, j=1:lon, k=1:lev)
        call assert(all(abs(outputs(i,j,k)%values() - outputs_elem_infer(i,j,k)%values()) < tolerance_2), &
          "all(openacc_offloading_outputs == outputs_elemental_infer)", &
          intrinsic_array_t(outputs(i,j,k)%values()-outputs_elem_infer(i,j,k)%values()))
      end do

      print *,"Performing batched inferences via intrinsic-array input and output"
      block 
        !associate(num_inputs => inference_engine%num_inputs())
          ! associate(inputs_batch => reshape( &
          !   [((((inputs(i,j,k)%values(), i=1,lat), j=1,lon), k=1,lev), n=1,num_inputs)], &
          !   shape=[lat,lon,lev,n] &
          ! ))
          call system_clock(t_start, clock_rate)
          output_components = batch_infer_(inference_engine,input_components)
          call system_clock(t_finish)
          print *,"Batch inference time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)
          !end associate
        !end associate
          do concurrent(i=1:lat, j=1:lon, k=1:lev)
            outputs(i,j,k) = tensor_t(output_components(i,j,k,:))
          end do
        
        !Batch inference test
        do concurrent(i=1:lat, j=1:lon, k=1:lev)
          call assert(all(abs(outputs(i,j,k)%values() - outputs_elem_infer(i,j,k)%values()) < tolerance), &
            "all(openmp_offloading_outputs == outputs_elemental_infer)")
        end do 
      end block
    end block
  end block

  contains
  function batch_infer_(self, inputs) result(outputs)
    implicit none
    class(inference_engine_t), intent(in) :: self
    real, intent(in) :: inputs(:,:,:,:)
    real, allocatable :: outputs(:,:,:,:)

    real(rkind), allocatable :: a(:,:,:,:,:)
    integer, parameter :: input_layer = 0
    integer :: output_layer
    integer k, l
    integer :: lat, lon, lev
    real(rkind), allocatable :: w(:,:,:), b(:,:)
    integer, allocatable :: n(:)

    lat = size(inputs,1)
    lon = size(inputs,2)  
    lev = size(inputs,3)
    w = self%weights_
    b = self%biases_
    n = self%nodes_
    output_layer = ubound(n,1)


    !call assert_consistency(self)

    !associate(w => self%weights_, b => self%biases_, n => self%nodes_, output_layer => ubound(self%nodes_,1))
      !associate(lat => size(inputs,1), lon => size(inputs,2), lev => size(inputs,3))

       allocate(a(lat, lon, lev, maxval(n), input_layer:output_layer))

       a(:,:,:,1:n(input_layer),input_layer) = inputs

       feed_forward: &
       do l = input_layer+1, output_layer
        block
          integer i, j, k
          !$omp parallel do default(none) shared(a,w,b,n,l,lat,lon,lev,self) schedule(static,1)
          do j=1,lon
            do k=1,lev
              do i=1,lat
              associate(z => matmul(w(1:n(l),1:n(l-1),l), a(i,j,k,1:n(l-1),l-1)) + b(1:n(l),l))
                a(i,j,k,1:n(l),l) = self%activation_strategy_%activation(z)
              end associate
              end do
            end do
          end do
          !$omp end parallel do
         end block
       end do feed_forward
      outputs = a(:,:,:,1:n(output_layer), output_layer)
    !   outputs = tensor_t(a(1:n(output_layer), output_layer))
       !end associate
   !end associate
    
  end function
  

  function matrix_vector_multiplication(m1, m2) result(m3)
    real(rkind), intent(in) :: m1(:,:), m2(:)
    real(rkind), dimension(size(m1,1)) :: m3
    integer :: i, j
    !!$omp target teams distribute parallel do map(from:m3) map(to:m1,m2) shared(m1,m2,m3)
    do i = 1, size(m1,1)
      m3(i) = 0
      do j = 1, size(m1,2)
        m3(i) = m3(i) + m1(i,j) * m2(j)
      end do
    end do
    !!$omp end target teams distribute parallel do
  end function
end program
