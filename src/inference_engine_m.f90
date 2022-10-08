module inference_engine_m
  !! Define an abstraction that supports inference operationsn on a neural network
  implicit none

  private
  public :: inference_engine_t, activation_function

  abstract interface
    
    pure function activation_function(x) result(y)
      real, intent(in) :: x
      real y
    end function

  end interface

  type inference_engine_t
    !! Encapsulate the minimal information needed to performance inference
    private
    real, allocatable :: input_weights_(:,:)    ! weights applied to go from the inputs to first hidden layer
    real, allocatable :: hidden_weights_(:,:,:) ! weights applied to go from one hidden layer to the next
    real, allocatable :: output_weights_(:,:)   ! weights applied to go from the final hidden layer to the outputs
    real, allocatable :: biases_(:,:)           ! neuronal offsets for each hidden layer
    procedure(activation_function), pointer, nopass :: activation_
  contains
    generic :: inference_engine_t => read_weights
    procedure :: read_weights
    procedure :: infer
  end type

  interface inference_engine_t

    pure module function construct(input_weights, hidden_weights, output_weights, biases, activation) result(inference_engine)
      implicit none
      real, intent(in), dimension(:,:) :: input_weights, output_weights, biases
      real, intent(in) :: hidden_weights(:,:,:)
      procedure(activation_function), intent(in), pointer :: activation
      type(inference_engine_t) inference_engine
    end function

  end interface

  interface

    module subroutine read_weights(self, file_name)
      implicit none
      class(inference_engine_t), intent(out) :: self
      character(len=*), intent(in) :: file_name
    end subroutine

    pure module function infer(self, input) result(output)
      implicit none
      class(inference_engine_t), intent(in) :: self
      real, intent(in) :: input(:)
      real, allocatable :: output(:)
    end function

  end interface

end module inference_engine_m