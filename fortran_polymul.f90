!-----------------------------------------------------------------------
subroutine polymul(n1, n2, p1, p2, p3)
! Calculate pile-up according to Barradas & Reis XRS 35 (2006).
!  Tw     shapping time (s)
!  Tp     time required for the pulse to reach the maximum value
!  time   total pulse time t (s)
!  Nt     total pulse rate
!  Tpulse = f*Tw, f depends on the amplifier (ref f=2)
!-----------------------------------------------------------------------
    implicit none

    integer, parameter :: dp = kind(1.d0)
    integer, intent(in) :: n1, n2
    real(dp), dimension(n1), intent(in) :: p1
    real(dp), dimension(n2), intent(in) :: p2
    real(dp), dimension(n1+n2-1), intent(out) :: p3
    integer :: i1, i2

    p3 = 0.0

    do i1 = 1, n1
        do i2 = 1, n2
            p3(i1+i2-1) = p3(i1+i2-1) + p1(i1) * p2(i2)
        end do
    end do

end subroutine polymul
