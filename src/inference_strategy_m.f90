module inference_strategy_m
  implicit none

  private
  public :: inference_strategy_t, activation_interface

  type, abstract :: inference_strategy_t
  contains
     procedure(infer_interface), nopass, deferred :: infer
  end type

  abstract interface

    pure function activation_interface(x) result(y)
      implicit none
      real, intent(in) :: x
      real y
    end function

    pure function infer_interface(neurons_per_layer, num_layers, num_outputs, &
      input, input_weights, hidden_weights, biases, output_biases, output_weights, activation)  result(output)
      import activation_interface
      implicit none
      integer, intent(in) :: neurons_per_layer, num_layers, num_outputs
      real, intent(in)  :: input(:)
      real, intent(in), allocatable :: input_weights(:,:)    !! weights applied to go from the inputs to first hidden layer
      real, intent(in), allocatable :: hidden_weights(:,:,:) !! weights applied to go from one hidden layer to the next
      real, intent(in), allocatable :: output_weights(:,:)   !! weights applied to go from the final hidden layer to the outputs
      real, intent(in), allocatable :: output_biases(:)      !! neuronal offsets applied to outputs
      real, intent(in), allocatable :: biases(:,:)           !! neuronal offsets for each hidden layer
      procedure(activation_interface), pointer, intent(in) :: activation
      real, allocatable :: output(:)
    end function

  end interface

end module inference_strategy_m
