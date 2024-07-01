! program print_training_configuration
!   !! Demonstrate how to construct and print a training_configuration_t object
!   use inference_engine_m, only : training_configuration_t, hyperparameters_t, network_configuration_t
!   use sourcery_m, only : file_t
!   implicit none

!   associate(training_configuration => training_configuration_t( &
!     hyperparameters_t(mini_batches=10, learning_rate=1.5, optimizer = "adam"), &
!     network_configuration_t(skip_connections=.false., nodes_per_layer=[2,72,2], activation_name="sigmoid") &
!   ))
!     associate(json_file => file_t(training_configuration%to_json()))
!       call json_file%write_lines()
!     end associate
!   end associate
! end program
