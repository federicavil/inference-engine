! Copyright (c), The Regents of the University of California
! Terms of use are as specified in LICENSE.txt
module activation_strategy_m
  use kind_parameters_m, only : rkind
  implicit none

  private
  public :: activation_strategy_t
  public :: activation_i

  type, abstract :: activation_strategy_t
  contains
     procedure(activation_i), nopass, deferred :: activation
  end type

  abstract interface

    elemental function activation_i(x) result(y)
      import rkind
      implicit none
      real(rkind), intent(in) :: x
      real(rkind) y
    end function

  end interface

end module activation_strategy_m
