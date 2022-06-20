program lithiumization
    implicit none
    character(:), allocatable :: denfile, espfile, tharg, dummy
    integer, allocatable :: element(:)
    real(8), allocatable :: position(:, :), den(:, :, :), esp(:, :, :)
    real(8) :: threshold = 0.001, origin(3), delta(3, 3), minesp = 0d0
    integer :: natoms, npts(3), len, u, i, j, k, mini = 0, minj = 0, mink = 0
    character(2), parameter :: symbol(118) = (/ 'H ', 'He', &
        'Li', 'Be', 'B ', 'C ', 'N ', 'O ', 'F ', 'Ne', &
        'Na', 'Mg', 'Al', 'Si', 'P ', 'S ', 'Cl', 'Ar', &
        'K ', 'Ca', 'Sc', 'Ti', 'V ', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', &
        'Rb', 'Sr', 'Y ', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I ', 'Xe', &
        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', &
        'Ta', 'W ', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', &
        'Pa', 'U ', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', &
        'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og' /)

    ! print title
    print *
    print '(a)', '+----------------------------------------------------------+'
    print '(a)', '| Fortran Lithiumization Program 1.1                       |'
    print '(a)', '| Copyright (c) 2020 Hiroyuki Matsui. All rights reserved. |'
    print '(a)', '+----------------------------------------------------------+'

    ! read command arguments
    call get_argument(1, denfile, len)
    if (len == 0) then
        print *
        print '(a)', 'usage: lithiumization den_cube_file esp_cube_file threshold'
        stop
    end if
    call get_argument(2, espfile, len)
    call get_argument(3, tharg, len)
    if (len > 0) then
        read (tharg, *) threshold
    end if

    ! read density file
    u = get_free_file_unit()
    open(u, file=denfile, status='old')
    read(u, *)
    read(u, *)
    read(u, *) natoms, origin(:)
    allocate(element(natoms), position(3, natoms))
    read(u, *) npts(1), delta(:, 1)
    read(u, *) npts(2), delta(:, 2)
    read(u, *) npts(3), delta(:, 3)
    allocate(den(npts(3), npts(2), npts(1)))
    do i = 1, natoms
        read(u, *) element(i), dummy, position(:, i)
    end do
    read(u, *) den(:, :, :)
    close(u)

    ! read espsity file
    u = get_free_file_unit()
    open(u, file=espfile, status='old')
    read(u, *)
    read(u, *)
    read(u, *) natoms, origin(:)
    read(u, *) npts(1), delta(:, 1)
    read(u, *) npts(2), delta(:, 2)
    read(u, *) npts(3), delta(:, 3)
    allocate(esp(npts(3), npts(2), npts(1)))
    do i = 1, natoms
        read(u, *)
    end do
    read(u, *) esp(:, :, :)
    close(u)

    ! convert unit from Bohr to Angstrom
    origin(:) = origin(:) * 5.29177210903d-1
    delta(:, :) = delta(:, :) * 5.29177210903d-1
    position(:, :) = position(:, :) * 5.29177210903d-1

    ! calculate potential minimum
    do i = 1, npts(1)
        do j = 1, npts(2)
            do k = 1, npts(3)
                if (den(k, j, i) < threshold .and. esp(k, j, i) < minesp) then
                    mini = i
                    minj = j
                    mink = k
                    minesp = esp(k, j, i)
                end if
            end do
        end do
    end do

    ! print results
    print *
    print '(a20, 1(x, a))', 'Density Cube File:', denfile
    print '(a20, 1(x, a))', 'ESP Cube File:', espfile
    print '(a20, 1(x, f13.8))', 'Threshold:', threshold
    print '(a20, 1(x, f13.8))', 'Minimum ESP:', minesp
    print '(a20, 3(x, i13))', 'Position (index):', mini, minj, mink
    print '(a20, 3(x, f13.8))', 'Position (Angstrom):', matmul(delta(:, :), (/ mini, minj, mink /)) + origin(:)
    print *
    print '(a)', 'For Pasting in GJF File:'
    do i = 1, natoms
        print '(x, a2, 3(x, f13.8))', symbol(element(i)), position(:, i)
    end do
    print '(x, a2, 3(x, f13.8))', 'Li', matmul(delta(:, :), (/ mini, minj, mink /)) + origin(:)

contains

    function get_free_file_unit()
        integer :: get_free_file_unit
        integer :: u, iostat
        logical :: opened
        do u = 10, 100
            inquire (unit=u, opened=opened, iostat=iostat)
            if (iostat/=0) cycle
            if (.not. opened) exit
        end do
        get_free_file_unit = u
    end

    subroutine get_argument(i, arg, len)
        integer, intent(in) :: i
        character(:), intent(inout), allocatable :: arg
        integer, intent(out) :: len

        call get_command_argument(i, length=len)
        if (allocated(arg)) then
            deallocate (arg)
        end if
        allocate (character(len) :: arg)
        call get_command_argument(i, arg)
    end subroutine get_argument
end program