module common

    implicit none

    ! mathematical constants
    real(8), parameter :: const_pi = atan(1d0)*4d0
    real(8), parameter :: degree_to_radian = const_pi/180d0
    real(8), parameter :: radian_to_degree = 1d0/degree_to_radian

    ! physical constants
    real(8), parameter :: const_h = 6.62607015d-34 ! planck constant
    real(8), parameter :: const_hbar = 6.62607015d-34/(2d0*const_pi) ! planck constant
    real(8), parameter :: const_k_b = 1.380649d-23 ! Boltzmann constant
    real(8), parameter :: const_n_a = 6.02214076d23 ! Avogadro constant
    real(8), parameter :: const_e = 1.602176634d-19 ! elementary charge
    real(8), parameter :: const_c = 2.99792458d8 ! speed of light
    real(8), parameter :: const_epsilon0 = 8.8541878128d-12 ! permittivity of vacuum

    ! element symbols
    character(2), parameter :: const_element_symbol(0:118) = (/ 'X ', 'H ', 'He', 'Li', 'Be', 'B ', 'C ', 'N ', 'O ', 'F ', &
        'Ne', 'Na', 'Mg', 'Al', 'Si', 'P ', 'S ', 'Cl', 'Ar', 'K ', 'Ca', 'Sc', 'Ti', 'V ', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', &
        'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y ', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', &
        'Cd', 'In', 'Sn', 'Sb', 'Te', 'I ', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', &
        'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W ', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', &
        'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U ', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', &
        'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og' /)

contains

    pure function vec_length(a)
        real(8), intent(in) :: a(:)
        real(8) :: vec_length
        vec_length = sqrt(sum(a*a))
    end

    pure function vec_length_squared(a)
        real(8), intent(in) :: a(:)
        real(8) :: vec_length_squared
        vec_length_squared = sum(a*a)
    end

    pure function vec_angle(a, b)
        real(8), intent(in) :: a(:), b(:)
        real(8) :: vec_angle
        vec_angle = acos(max(-1d0, min(dot_product(a,b)/sqrt(sum(a*a)*sum(b*b)), 1d0)))
    end

    pure function vec_cross_product(a, b)
        real(8), intent(in) :: a(3), b(3)
        real(8) :: vec_cross_product(3)
        vec_cross_product(1) = a(2)*b(3) - a(3)*b(2)
        vec_cross_product(2) = a(3)*b(1) - a(1)*b(3)
        vec_cross_product(3) = a(1)*b(2) - a(2)*b(1)
    end

    pure function mat_determinant(a)
        real(8), intent(in) :: a(3, 3)
        real(8) :: mat_determinant
        mat_determinant = &
        a(1, 1)*(a(2,2)*a(3,3)-a(2,3)*a(3,2)) + &
        a(1, 2)*(a(2,3)*a(3,1)-a(2,1)*a(3,3)) + &
        a(1, 3)*(a(2,1)*a(3,2)-a(2,2)*a(3,1))
    end

    pure function mat_determinant_derivative(a) result (d)
        real(8), intent(in) :: a(3, 3)
        real(8) :: d(3, 3)
        ! mat_determinant = a(1, 1)*(a(2,2)*a(3,3)-a(2,3)*a(3,2)) + a(1, 2)*(a(2,3)*a(3,1)-a(2,1)*a(3,3)) + &
        ! a(1, 3)*(a(2,1)*a(3,2)-a(2,2)*a(3,1))
        d(1, 1) = a(2, 2)*a(3, 3) - a(2, 3)*a(3, 2)
        d(1, 2) = a(2, 3)*a(3, 1) - a(2, 1)*a(3, 3)
        d(1, 3) = a(2, 1)*a(3, 2) - a(2, 2)*a(3, 1)
        d(2, 1) = a(3, 2)*a(1, 3) - a(3, 3)*a(1, 2)
        d(2, 2) = a(3, 3)*a(1, 1) - a(3, 1)*a(1, 3)
        d(2, 3) = a(3, 1)*a(1, 2) - a(3, 2)*a(1, 1)
        d(3, 1) = a(1, 2)*a(2, 3) - a(1, 3)*a(2, 2)
        d(3, 2) = a(1, 3)*a(2, 1) - a(1, 1)*a(2, 3)
        d(3, 3) = a(1, 1)*a(2, 2) - a(1, 2)*a(2, 1)
    end

    pure function mat_inverse(a)
        real(8), intent(in) :: a(3, 3)
        real(8) :: mat_inverse(3, 3)
        real(8) :: inv_det
        inv_det = 1d0/mat_determinant(a)
        mat_inverse(1, 1) = inv_det*(a(2,2)*a(3,3)-a(2,3)*a(3,2))
        mat_inverse(1, 2) = inv_det*(a(1,3)*a(3,2)-a(1,2)*a(3,3))
        mat_inverse(1, 3) = inv_det*(a(1,2)*a(2,3)-a(1,3)*a(2,2))
        mat_inverse(2, 1) = inv_det*(a(2,3)*a(3,1)-a(2,1)*a(3,3))
        mat_inverse(2, 2) = inv_det*(a(1,1)*a(3,3)-a(1,3)*a(3,1))
        mat_inverse(2, 3) = inv_det*(a(1,3)*a(2,1)-a(1,1)*a(2,3))
        mat_inverse(3, 1) = inv_det*(a(2,1)*a(3,2)-a(2,2)*a(3,1))
        mat_inverse(3, 2) = inv_det*(a(1,2)*a(3,1)-a(1,1)*a(3,2))
        mat_inverse(3, 3) = inv_det*(a(1,1)*a(2,2)-a(1,2)*a(2,1))
    end

    pure function ltc_length_a(plv)
        real(8), intent(in) :: plv(3, 3)
        real(8) :: ltc_length_a
        ltc_length_a = vec_length(plv(1:3,1))
    end

    pure function ltc_length_b(plv)
        real(8), intent(in) :: plv(3, 3)
        real(8) :: ltc_length_b
        ltc_length_b = vec_length(plv(1:3,2))
    end

    pure function ltc_length_c(plv)
        real(8), intent(in) :: plv(3, 3)
        real(8) :: ltc_length_c
        ltc_length_c = vec_length(plv(1:3,3))
    end

    pure function ltc_lengths(plv)
        real(8), intent(in) :: plv(3, 3)
        real(8) :: ltc_lengths(3)
        ltc_lengths(1) = ltc_length_a(plv)
        ltc_lengths(2) = ltc_length_b(plv)
        ltc_lengths(3) = ltc_length_c(plv)
    end

    pure function ltc_angle_alpha(plv)
        real(8), intent(in) :: plv(3, 3)
        real(8) :: ltc_angle_alpha
        ltc_angle_alpha = vec_angle(plv(1:3,2), plv(1:3,3))*radian_to_degree
    end

    pure function ltc_angle_beta(plv)
        real(8), intent(in) :: plv(3, 3)
        real(8) :: ltc_angle_beta
        ltc_angle_beta = vec_angle(plv(1:3,3), plv(1:3,1))*radian_to_degree
    end

    pure function ltc_angle_gamma(plv)
        real(8), intent(in) :: plv(3, 3)
        real(8) :: ltc_angle_gamma
        ltc_angle_gamma = vec_angle(plv(1:3,1), plv(1:3,2))*radian_to_degree
    end

    pure function ltc_angles(plv)
        real(8), intent(in) :: plv(3, 3)
        real(8) :: ltc_angles(3)
        ltc_angles(1) = ltc_angle_alpha(plv)
        ltc_angles(2) = ltc_angle_beta(plv)
        ltc_angles(3) = ltc_angle_gamma(plv)
    end

    pure function ltc_constants(plv)
        real(8), intent(in) :: plv(3, 3)
        real(8) :: ltc_constants(6)
        ltc_constants(1) = ltc_length_a(plv)
        ltc_constants(2) = ltc_length_b(plv)
        ltc_constants(3) = ltc_length_c(plv)
        ltc_constants(4) = ltc_angle_alpha(plv)
        ltc_constants(5) = ltc_angle_beta(plv)
        ltc_constants(6) = ltc_angle_gamma(plv)
    end

    pure function ltc_volume(plv)
        real(8), intent(in) :: plv(3, 3)
        real(8) :: ltc_volume
        ltc_volume = abs(mat_determinant(plv))
    end

    ! @ ---- plane distances ----

    ! bc�ʊԂ̋���
    pure function ltc_bc_plane_distance(plv) result (d)
        real(8), intent(in) :: plv(3, 3)
        real(8) :: d
        real(8) :: inv_plv(3, 3)
        inv_plv = mat_inverse(plv)
        d = 1d0/vec_length(inv_plv(1,1:3))
    end

    ! ca�ʊԂ̋���
    pure function ltc_ca_plane_distance(plv) result (d)
        real(8), intent(in) :: plv(3, 3)
        real(8) :: d
        real(8) :: inv_plv(3, 3)
        inv_plv = mat_inverse(plv)
        d = 1d0/vec_length(inv_plv(2,1:3))
    end

    ! ab�ʊԂ̋���
    pure function ltc_ab_plane_distance(plv) result (d)
        real(8), intent(in) :: plv(3, 3)
        real(8) :: d
        real(8) :: inv_plv(3, 3)
        inv_plv = mat_inverse(plv)
        d = 1d0/vec_length(inv_plv(3,1:3))
    end

    ! @ ---- conversions ----

    ! convert a, b, c, alpha, beta, gamma into ax, bx, by, cx, cy, cz
    pure subroutine ltc_vectors(lc, plv)
        real(8), intent(in) :: lc(6)
        real(8), intent(out) :: plv(3, 3)
        real(8) :: alpha, beta, gamma, cosg, sing, cosb, cosa
        alpha = lc(4)*degree_to_radian
        beta = lc(5)*degree_to_radian
        gamma = lc(6)*degree_to_radian
        cosg = cos(gamma)
        sing = sin(gamma)
        cosb = cos(beta)
        cosa = cos(alpha)
        plv(1, 1) = lc(1)
        plv(2, 1) = 0d0
        plv(3, 1) = 0d0
        plv(1, 2) = lc(2)*cosg
        plv(2, 2) = lc(2)*sing
        plv(3, 2) = 0d0
        plv(1, 3) = lc(3)*cosb
        plv(2, 3) = lc(3)*(cosa-cosb*cosg)/sing
        plv(3, 3) = sqrt(lc(3)**2-plv(1,3)**2-plv(2,3)**2)
    end

    ! �t�i�q�x�N�g���̌v�Z
    pure subroutine ltc_reciprocal(plv, pkv)
        real(8), intent(in) :: plv(3, 3)
        real(8), intent(out) :: pkv(3, 3)
        pkv = 2d0*const_pi*mat_inverse(transpose(plv))
    end

    ! ���q�ʒu(x,y,z)���܂܂��Z���̃C���f�b�N�X
    ! pure function ltc_cell(pos, inv_plv)
    ! real(8), intent(in) :: pos(3), inv_plv(6)
    ! integer :: ltc_cell(3)
    ! ltc_cell(1:3) = floor(umat3_times_vec3(inv_plv(1:6),pos(1:3)))
    ! end function ltc_cell

    ! �Q�ƃZ�����̓����Ȉʒu
    ! pure function ltc_pos_in_reference_cell(pos, plv, pkv)
    ! real(8), intent(in) :: pos(3), plv(6), pkv(6)
    ! real(8) :: ltc_pos_in_reference_cell(3)
    ! ltc_pos_in_reference_cell(1:3) = pos - umat3_times_vec3(plv(1:6), dble(floor(umat3_times_vec3(pkv(1:6),pos(1:3)))))
    ! end function ltc_pos_in_reference_cell

    ! pure function ltc_shortest_equivalent(pos, plv, inv_plv) result (shortest)
    ! real(8), intent(in) :: pos(3), plv(6), inv_plv(6)
    ! real(8) :: shortest(3)
    ! ! integer :: cell(3)
    ! shortest = pos - umat3_times_vec3(plv(1:6), dble(nint(umat3_times_vec3(inv_plv(1:6),pos(1:3)))))
    ! ! cell(1:3) = nint(ltc_convert(pos, inv_plv))
    ! ! shortest(1) = cart(1) - cell(3)*plv(4) - cell(2)*plv(2) - cell(1)*plv(1)
    ! ! shortest(2) = cart(2) - cell(3)*plv(5) - cell(2)*plv(3)
    ! ! shortest(3) = cart(3) - cell(3)*plv(6)
    ! end function ltc_shortest_equivalent

    ! @ ---- lattice reduction ----

    ! lattice reduction
    pure subroutine ltc_lattice_reduction(plv, trans)
        real(8), intent(inout) :: plv(3, 3) ! primitive lattice vectors
        integer, intent(out) :: trans(3, 3) ! transformation matrix

        integer :: i, j, k, trial, pre(3), new(3)
        real(8) :: min2, r2, cos1, cos2, cos3
        logical :: updated

        integer, parameter :: max_trial = 100
        integer, parameter :: rng = 10 ! range for searching

        ! search the shortest lattice vector
        pre(1:3) = (/ 1, 0, 0 /)
        min2 = 1d9
        do trial = 1, max_trial
        updated = .false.
        do i = pre(1) - rng, pre(1) + rng
            do j = pre(2) - rng, pre(2) + rng
            do k = pre(3) - rng, pre(3) + rng
                if (i/=0 .or. j/=0 .or. k/=0) then
                r2 = vec_length_squared(matmul(plv,(/i,j,k/)))
                if (r2<min2) then
                    new(1:3) = (/ i, j, k /)
                    min2 = r2
                    updated = .true.
                end if
                end if
            end do
            end do
        end do
        if (updated) then
            pre(1:3) = new(1:3)
        else
            exit
        end if
        end do
        trans(1:3, 1) = pre(1:3)

        ! search the second shortest lattice vector
        pre(1:3) = (/ 0, 1, 0 /)
        min2 = 1d9
        do trial = 1, max_trial
        updated = .false.
        do i = pre(1) - rng, pre(1) + rng
            do j = pre(2) - rng, pre(2) + rng
            do k = pre(3) - rng, pre(3) + rng
                if (i/=0 .or. j/=0 .or. k/=0) then
                ! if not parallel to the shortest lattice vector
                if (sum(trans(1:3,1)**2)*(i**2+j**2+k**2)/=dot_product(trans(1:3,1),(/i,j,k/))**2) then
                    r2 = vec_length_squared(matmul(plv,(/i,j,k/)))
                    if (r2<min2) then
                    new(1:3) = (/ i, j, k /)
                    min2 = r2
                    updated = .true.
                    end if
                end if
                end if
            end do
            end do
        end do
        if (updated) then
            pre(1:3) = new(1:3)
        else
            exit
        end if
        end do
        trans(1:3, 2) = pre(1:3)

        ! search the third shortest lattice vector
        pre(1:3) = (/ 0, 0, 1 /)
        min2 = 1d9
        do trial = 1, max_trial
        updated = .false.
        do i = pre(1) - rng, pre(1) + rng
            do j = pre(2) - rng, pre(2) + rng
            do k = pre(3) - rng, pre(3) + rng
                if (i/=0 .or. j/=0 .or. k/=0) then
                ! if not parallel to the plane made by the shortest and the second shortest lattice vectors
                if (i*(trans(2,1)*trans(3,2)-trans(3,1)*trans(2,2))+j*(trans(3,1)*trans(1,2)-trans(1,1)*trans(3,2))+ &
                k*(trans(1,1)*trans(2,2)-trans(2,1)*trans(1,2))/=0d0) then
                    r2 = vec_length_squared(matmul(plv,(/i,j,k/)))
                    if (r2<min2) then
                    new(1:3) = (/ i, j, k /)
                    min2 = r2
                    updated = .true.
                    end if
                end if
                end if
            end do
            end do
        end do
        if (updated) then
            pre(1:3) = new(1:3)
        else
            exit
        end if
        end do
        trans(1:3, 3) = pre(1:3)

        ! calculate new lattice vectors
        plv = matmul(plv, trans)

        ! if left-hand system, invert all vectors
        if (dot_product(vec_cross_product(plv(1:3,1),plv(1:3,2)),plv(1:3,3))<0d0) then
        trans = -trans
        plv = -plv
        end if

        ! make type-I (all acute) or type-II (all non-acute)
        cos1 = dot_product(plv(1:3,2), plv(1:3,3))/sqrt(sum(plv(1:3,2)**2)*sum(plv(1:3,3)**2))
        cos2 = dot_product(plv(1:3,3), plv(1:3,1))/sqrt(sum(plv(1:3,3)**2)*sum(plv(1:3,1)**2))
        cos3 = dot_product(plv(1:3,1), plv(1:3,2))/sqrt(sum(plv(1:3,1)**2)*sum(plv(1:3,2)**2))
        if (cos1>1d-3) then
        if (cos2>1d-3) then
            if (cos3>1d-3) then
            ! ok
            else
            trans(1:3, 3) = -trans(1:3, 3)
            plv(1:3, 3) = -plv(1:3, 3)
            end if
        else
            if (cos3>1d-3) then
            trans(1:3, 2) = -trans(1:3, 2)
            plv(1:3, 2) = -plv(1:3, 2)
            else
            trans(1:3, 1) = -trans(1:3, 1)
            plv(1:3, 1) = -plv(1:3, 1)
            end if
        end if
        else
        if (cos2>1d-3) then
            if (cos3>1d-3) then
            trans(1:3, 1) = -trans(1:3, 1)
            plv(1:3, 1) = -plv(1:3, 1)
            else
            trans(1:3, 2) = -trans(1:3, 2)
            plv(1:3, 2) = -plv(1:3, 2)
            end if
        else
            if (cos3>1d-3) then
            trans(1:3, 3) = -trans(1:3, 3)
            plv(1:3, 3) = -plv(1:3, 3)
            else
            ! ok
            end if
        end if
        end if

    end

    subroutine cif_write(unit, dataname, n_atom, pos, plv, atom_num, adp)
        integer, intent(in) :: unit, n_atom
        character (*), intent(in) :: dataname
        real(8), intent(in) :: pos(3, n_atom), plv(3, 3)
        integer, intent(in) :: atom_num(n_atom)
        real(8), intent(in), optional :: adp(3, 3, n_atom)
        ! logical, intent(in) :: adp_enabled
        real(8) :: inv_plv(3, 3)
        integer :: i
        logical :: opened

        inquire (unit, opened=opened)
        if (opened) then

        ! inv_plv = inv_umat3(plv)
        inv_plv = mat_inverse(plv)

        write (unit, *) 'data_' // dataname
        write (unit, *) '_symmetry_cell_setting           triclinic'
        write (unit, *) '_symmetry_space_group_name_H-M   ''P 1'''
        write (unit, *) '_symmetry_Int_Tables_number      1'
        write (unit, *) '_space_group_name_Hall           ''P 1'''
        write (unit, *) 'loop_'
        write (unit, *) '_symmetry_equiv_pos_site_id'
        write (unit, *) '_symmetry_equiv_pos_as_xyz'
        write (unit, *) '1 x,y,z'
        write (unit, *) '_cell_length_a', vec_length(plv(:,1))
        write (unit, *) '_cell_length_b', vec_length(plv(:,2))
        write (unit, *) '_cell_length_c', vec_length(plv(:,3))
        write (unit, *) '_cell_angle_alpha', vec_angle(plv(:,2), plv(:,3))
        write (unit, *) '_cell_angle_beta ', vec_angle(plv(:,3), plv(:,1))
        write (unit, *) '_cell_angle_gamma', vec_angle(plv(:,1), plv(:,2))
        write (unit, *) '_cell_volume', abs(mat_determinant(plv))
        write (unit, *) 'loop_'
        write (unit, *) '_atom_site_label'
        write (unit, *) '_atom_site_type_symbol'
        write (unit, *) '_atom_site_fract_x'
        write (unit, *) '_atom_site_fract_y'
        write (unit, *) '_atom_site_fract_z'
        do i = 1, n_atom
            write (unit, *) i, const_element_symbol(atom_num(i)), matmul(inv_plv, pos(:,i))
        end do
        if (present(adp)) then
            write (unit, *) 'loop_'
            write (unit, *) '_atom_site_aniso_label'
            write (unit, *) '_atom_site_aniso_U_11'
            write (unit, *) '_atom_site_aniso_U_12'
            write (unit, *) '_atom_site_aniso_U_13'
            write (unit, *) '_atom_site_aniso_U_22'
            write (unit, *) '_atom_site_aniso_U_23'
            write (unit, *) '_atom_site_aniso_U_33'
            do i = 1, n_atom
            write (unit, *) i, adp(1, 1:3, i), adp(2, 2:3, i), adp(3, 3, i)
            end do
        end if
        write (unit, *)

        end if
    end

    pure subroutine ewald_update_lattice_vectors(plv, nli, li, lv)
        real(8), intent(in) :: plv(3, 3)
        integer, intent(in) :: nli, li(3, nli)
        real(8), intent(out) :: lv(3, nli)
        integer :: i
        do i = 1, nli
        lv(:, i) = matmul(plv(:,:), li(:,i))
        end do
    end

    ! �i�q�x�N�g�����X�g�̍X�V�i�x�N�g���̌����X�V�j
    ! �i��x�N�g���Ɣ��]�Ώ̂ȃx�N�g���͏����j
    pure subroutine ewald_refresh_lattice_vectors(cutoff_l, plv, nli, li, lv)
        real(8), intent(in) :: cutoff_l, plv(3, 3)
        integer, intent(out) :: nli
        integer, intent(inout), allocatable :: li(:, :)
        real(8), intent(inout), allocatable :: lv(:, :)
        integer :: capacity, i, j, k, cell_range(3)
        real(8) :: cutoff2, v(3)

        nli = 0
        capacity = size(lv, 2)
        cutoff2 = cutoff_l**2
        cell_range = ewald_cell_range_within_cutoff(cutoff_l, plv)

        do k = 1, cell_range(3)
        do j = -cell_range(2), cell_range(2)
            do i = -cell_range(1), cell_range(1)
            v = matmul(plv(:,:), (/i,j,k/) )
            if (sum(v(1:3)**2)<=cutoff2) then
                nli = nli + 1
                if (nli>capacity) then
                call double_size(capacity, li, lv)
                end if
                li(1:3, nli) = (/ i, j, k /)
                lv(1:3, nli) = v(1:3)
            end if
            end do
        end do
        end do

        do j = 1, cell_range(2)
        do i = -cell_range(1), cell_range(1)
            v = matmul(plv(:,:), (/i,j,0/) )
            if (sum(v(1:3)**2)<=cutoff2) then
            nli = nli + 1
            if (nli>capacity) then
                call double_size(capacity, li, lv)
            end if
            li(1:3, nli) = (/ i, j, 0 /)
            lv(1:3, nli) = v(1:3)
            end if
        end do
        end do

        do i = 1, cell_range(1)
        v = matmul(plv(:,:), (/i,0,0/) )
        if (sum(v(1:3)**2)<=cutoff2) then
            nli = nli + 1
            if (nli>capacity) then
            call double_size(capacity, li, lv)
            end if
            li(1:3, nli) = (/ i, 0, 0 /)
            lv(1:3, nli) = v(1:3)
        end if
        end do

    end

    ! @ hide
    ! �J�b�g�I�t���̋����܂ލŏ��̃Z���͈͂��擾
    pure function ewald_cell_range_within_cutoff(cutoff_l, plv) result (cell)
        real(8), intent(in) :: cutoff_l, plv(3, 3)
        integer :: cell(3)
        cell(1) = ceiling(cutoff_l/ltc_bc_plane_distance(plv))
        cell(2) = ceiling(cutoff_l/ltc_ca_plane_distance(plv))
        cell(3) = ceiling(cutoff_l/ltc_ab_plane_distance(plv))
    end

    ! @ hide
    ! �z��̃T�C�Y���Q�{�ɂ���
    pure subroutine double_size(capacity, li, lv)
        integer, intent(inout) :: capacity
        integer, intent(inout), allocatable :: li(:, :)
        real(8), intent(inout), allocatable :: lv(:, :)
        integer :: ibuf(3, capacity)
        real(8) :: rbuf(3, capacity)

        ibuf(1:3, 1:capacity) = li(1:3, 1:capacity)
        rbuf(1:3, 1:capacity) = lv(1:3, 1:capacity)
        deallocate (li, lv)
        allocate (li(3,capacity*2), lv(3,capacity*2))
        li(1:3, 1:capacity) = ibuf(1:3, 1:capacity)
        lv(1:3, 1:capacity) = rbuf(1:3, 1:capacity)

        capacity = capacity*2
    end

    ! @ ======== �G�l���M�[�Ƃ��̔����̌v�Z ========

    ! �G�l���M�[�Ƃ��̔����̌v�Z�i����ԁA�P��Z���j
    subroutine ewald_add_real_sum_in_single_cell(n, pos, rho_func, mode, energy, grad1, hess)
        integer, intent(in) :: n, mode
        real(8), intent(in) :: pos(3, n)
        real(8), intent(inout) :: energy
        real(8), intent(inout), optional :: grad1(3, n), hess(3, n, 3, n)
        interface
        subroutine rho_func(i, j, r2, mode, rho0, rho1, rho2)
            integer, intent(in) :: i, j, mode
            real(8), intent(in) :: r2
            real(8), intent(out) :: rho0, rho1, rho2
        end subroutine rho_func
        end interface

        integer :: i, j, s, t
        real(8) :: rij(3), rij2, rho0, rho1, rho2, g(3), h(3, 3)

        if (mode==0) then
        do i = 1, n
            do j = 1, i - 1
            call rho_func(i, j, sum((pos(1:3,i)-pos(1:3,j))**2), mode, rho0, rho1, rho2)
            energy = energy + rho0
            end do
        end do
        return
        end if

        do i = 1, n
        do j = 1, i - 1
            rij(1:3) = pos(1:3, i) - pos(1:3, j)
            rij2 = sum(rij*rij)
            call rho_func(i, j, rij2, mode, rho0, rho1, rho2)

            energy = energy + rho0

            ! calculate gradient
            if (mode/=0) then
            g(1:3) = rho1*rij(1:3)
            grad1(1:3, i) = grad1(1:3, i) + g(1:3)
            grad1(1:3, j) = grad1(1:3, j) - g(1:3)
            end if

            ! calculate hessian
            if (mode==2 .or. mode==-2) then
            do s = 1, 3
                do t = 1, 3
                h(s, t) = rho2*rij(s)*rij(t)
                if (s==t) then
                    h(s, t) = h(s, t) + rho1
                end if
                end do
            end do
            hess(1:3, i, 1:3, i) = hess(1:3, i, 1:3, i) + h(1:3, 1:3)
            hess(1:3, j, 1:3, j) = hess(1:3, j, 1:3, j) + h(1:3, 1:3)
            hess(1:3, i, 1:3, j) = hess(1:3, i, 1:3, j) - h(1:3, 1:3)
            hess(1:3, j, 1:3, i) = hess(1:3, j, 1:3, i) - h(1:3, 1:3)
            end if

        end do
        end do
    end

    ! �G�l���M�[�Ƃ��̔����̌v�Z�i����ԁA�S�Z���j
    subroutine ewald_add_real_sum(n, pos, nli, li, lv, rho_func, mode, energy, grad1, hess, grad2)
        integer, intent(in) :: n, nli, li(3, nli), mode
        real(8), intent(in) :: pos(3, n), lv(3, nli)
        real(8), intent(inout) :: energy, grad1(3, n), grad2(3, 3), hess(3, n, 3, n)
        interface
        subroutine rho_func(i, j, r2, mode, rho0, rho1, rho2)
            integer, intent(in) :: i, j, mode
            real(8), intent(in) :: r2
            real(8), intent(out) :: rho0, rho1, rho2
        end subroutine rho_func
        end interface

        integer :: i, j, l, s, t
        real(8) :: rijl(3), rijl2, rho0, rho1, rho2, g(3), h(3, 3)

        if (mode==0) then
        !$omp parallel shared (n, nli, pos, lv, energy, mode) private (l, i, j, rho0, rho1, rho2)
        !$omp do reduction (+:energy)
        do l = 1, nli
            do i = 1, n
            do j = 1, n
                call rho_func(i, j, sum((pos(1:3,i)-pos(1:3,j)+lv(1:3,l))**2), mode, rho0, rho1, rho2)
                energy = energy + rho0
            end do
            end do
        end do
        !$omp end do
        !$omp end parallel

        call ewald_add_real_sum_in_single_cell(n, pos, rho_func, mode, energy, grad1, hess)

        return
        end if

        !$omp parallel shared (n, nli, pos, li, lv, energy, grad1, hess, grad2, mode) &
        !$omp private (i, j, l, s, t, rijl, rijl2, rho0, rho1, rho2, g, h)
        !$omp do reduction (+:energy, grad1, hess, grad2)
        do l = 1, nli
        do i = 1, n
            do j = 1, n
            rijl(1:3) = pos(1:3, i) - pos(1:3, j) + lv(1:3, l)
            rijl2 = sum(rijl*rijl)
            call rho_func(i, j, rijl2, mode, rho0, rho1, rho2)
            energy = energy + rho0

            if (mode/=0) then
                g(1:3) = rho1*rijl(1:3)
                grad1(1:3, i) = grad1(1:3, i) + g(1:3)
                grad1(1:3, j) = grad1(1:3, j) - g(1:3)
            end if
            if (mode<0) then
                grad2(:, 1) = grad2(:, 1) + g(:)*li(1, l)
                grad2(:, 2) = grad2(:, 2) + g(:)*li(2, l)
                grad2(:, 3) = grad2(:, 3) + g(:)*li(3, l)
            end if
            if (mode==2 .or. mode==-2) then
                do s = 1, 3
                do t = 1, 3
                    h(s, t) = rho2*rijl(s)*rijl(t)
                    if (s==t) then
                    h(s, t) = h(s, t) + rho1
                    else
                    end if
                end do
                end do
                hess(1:3, i, 1:3, i) = hess(1:3, i, 1:3, i) + h(1:3, 1:3)
                hess(1:3, j, 1:3, j) = hess(1:3, j, 1:3, j) + h(1:3, 1:3)
                hess(1:3, i, 1:3, j) = hess(1:3, i, 1:3, j) - h(1:3, 1:3)
                hess(1:3, j, 1:3, i) = hess(1:3, j, 1:3, i) - h(1:3, 1:3)
            end if
            end do
        end do
        end do
        !$omp end do
        !$omp end parallel

        call ewald_add_real_sum_in_single_cell(n, pos, rho_func, mode, energy, grad1, hess)

    end

    ! �G�l���M�[�Ƃ��̔����̌v�Z�i�t��ԁA�S�Z���j
    subroutine ewald_add_reciprocal_sum(n, pos, plv, nkv, kv, c, phi_func, tau, mode, energy, grad1, hess, grad2)
        integer, intent(in) :: n, nkv, mode
        real(8), intent(in) :: pos(3, n), plv(3, 3), kv(3, nkv), c(n, n), tau
        real(8), intent(inout) :: energy
        real(8), intent(inout), optional :: grad1(3, n), hess(3, n, 3, n), grad2(3, 3)
        interface
        subroutine phi_func(k2, mode, phi0, phi1)
            integer, intent(in) :: mode
            real(8), intent(in) :: k2
            real(8), intent(out) :: phi0, phi1
        end subroutine phi_func
        end interface

        integer :: i, j, k, s, t
        real(8) :: volume, inv_volume
        real(8) :: pkv(3, 3), k2, phi0, phi1
        real(8) :: pf_grad(3), pf_hess(3, 3), kkk(3, 3), kk(3, 3, 3), dvdplv(3, 3)
        real(8) :: rij(3), dot, c_cos, sum_c_cos, c_sin, sum_rij_c_sin(3)
        real(8) :: g(3), h(3, 3)
        real(8), parameter :: pi_times_2 = const_pi*2d0

        volume = abs(mat_determinant(plv))
        inv_volume = 1d0/volume

        if (mode==0) then
        ! non-zero k term
        !$omp parallel shared (n, nkv, pos, kv, mode, energy, c, inv_volume) &
        !$omp private (i, j, k, k2, phi0, phi1, sum_c_cos)
        !$omp do reduction (+:energy)
        do k = 1, nkv
            k2 = sum(kv(1:3,k)*kv(1:3,k))
            call phi_func(k2, mode, phi0, phi1)
            sum_c_cos = 0d0
            do i = 1, n
            sum_c_cos = sum_c_cos + 5d-1*c(i, i)
            do j = 1, i - 1
                sum_c_cos = sum_c_cos + c(i, j)*cos(dot_product(kv(1:3,k),pos(1:3,i)-pos(1:3,j)))
            end do
            end do
            energy = energy + 2*inv_volume*phi0*sum_c_cos
        end do
        !$omp end do
        !$omp end parallel

        ! zero k term
        call phi_func(0d0, mode, phi0, phi1)
        energy = energy + 5d-1*inv_volume*phi0*sum(c(1:n,1:n))

        ! self correction term
        sum_c_cos = 0d0
        do i = 1, n
            sum_c_cos = sum_c_cos + c(i, i)
        end do
        energy = energy - tau*sum_c_cos

        return
        end if

        if (mode<0) then
        dvdplv(:, :) = mat_determinant_derivative(plv)
        call ltc_reciprocal(plv, pkv)
        end if

        ! non-zero k term
        !$omp parallel shared(n, nkv, pos, kv, pkv, mode, energy, grad1, hess, grad2, inv_volume, dvdplv) &
        !$omp private(i, j, k, k2, s, t, phi0, phi1, pf_grad, pf_hess, kkk, kk, g, h, dot, c_cos, c_sin, &
        !$omp sum_c_cos, rij, sum_rij_c_sin)
        !$omp do reduction(+:energy, grad1, hess, grad2)
        do k = 1, nkv

        k2 = sum(kv(1:3,k)*kv(1:3,k))
        call phi_func(k2, mode, phi0, phi1)

        if (mode/=0) then
            pf_grad(1:3) = -2*inv_volume*phi0*kv(1:3, k)
        end if

        if (mode==2 .or. mode==-2) then
            do s = 1, 3
            do t = 1, 3
                pf_hess(s, t) = -2*inv_volume*phi0*kv(s, k)*kv(t, k)
            end do
            end do
        end if

        if (mode<0) then
            kk(1, :, :) = kv(1, k)*transpose(pkv)
            kk(2, :, :) = kv(2, k)*transpose(pkv)
            kk(3, :, :) = kv(3, k)*transpose(pkv)
            kkk(:, :) = kk(:, :, 1)*kv(1, k) + kk(:, :, 2)*kv(2, k) + kk(:, :, 3)*kv(3, k)
            kk(:, :, :) = kk(:, :, :)/(const_pi*volume)*phi0
            kkk(:, :) = -2*inv_volume**2*(phi0*dvdplv(:,:)+volume/pi_times_2*phi1*kkk(:,:))
        end if

        sum_c_cos = 0d0
        sum_rij_c_sin(1:3) = 0d0

        do i = 1, n

            sum_c_cos = sum_c_cos + 5d-1*c(i, i)

            do j = 1, i - 1

            rij(1:3) = pos(1:3, i) - pos(1:3, j)
            dot = dot_product(kv(1:3,k), rij(1:3))
            c_cos = c(i, j)*cos(dot)
            sum_c_cos = sum_c_cos + c_cos

            ! calculate gradient
            if (mode/=0) then
                c_sin = c(i, j)*sin(dot)
                g(1:3) = pf_grad*c_sin
                grad1(1:3, i) = grad1(1:3, i) + g(1:3)
                grad1(1:3, j) = grad1(1:3, j) - g(1:3)
            end if

            ! calculate hessian
            if (mode==2 .or. mode==-2) then
                h(1:3, 1:3) = pf_hess*c_cos
                hess(1:3, i, 1:3, i) = hess(1:3, i, 1:3, i) + h(1:3, 1:3)
                hess(1:3, j, 1:3, j) = hess(1:3, j, 1:3, j) + h(1:3, 1:3)
                hess(1:3, i, 1:3, j) = hess(1:3, i, 1:3, j) - h(1:3, 1:3)
                hess(1:3, j, 1:3, i) = hess(1:3, j, 1:3, i) - h(1:3, 1:3)
            end if

            ! for calculation of gradient with lattice
            if (mode<0) then
                sum_rij_c_sin(1:3) = sum_rij_c_sin(1:3) + rij(1:3)*c_sin
            end if

            end do
        end do

        energy = energy + 2*inv_volume*phi0*sum_c_cos

        if (mode<0) then
            grad2(:, :) = grad2(:, :) + kkk(:, :)*sum_c_cos
            grad2(:, :) = grad2(:, :) + kk(:, :, 1)*sum_rij_c_sin(1) + kk(:, :, 2)*sum_rij_c_sin(2) + &
            kk(:, :, 3)*sum_rij_c_sin(3)
        end if

        end do
        !$omp end do
        !$omp end parallel

        ! zero k term
        call phi_func(0d0, mode, phi0, phi1)
        sum_c_cos = sum(c(1:n,1:n))
        energy = energy + 5d-1*inv_volume*phi0*sum_c_cos
        if (mode<0) then
        grad2(:, :) = grad2(:, :) - 5d-1*inv_volume**2*phi0*sum_c_cos*dvdplv(:, :)
        end if

        ! self correction term
        sum_c_cos = 0d0
        do i = 1, n
        sum_c_cos = sum_c_cos + c(i, i)
        end do
        energy = energy - tau*sum_c_cos

    end

    ! �G�l���M�[�Ƃ��̔����̌v�Z
    subroutine ewald_add_ewald_sum(n, pos, plv, c, nli, li, lv, nkv, kv, rho_func, phi_func, tau, mode, energy, grad1, &
        hess, grad2)
        integer, intent(in) :: n, nli, li(3, nli), nkv, mode
        real(8), intent(in) :: pos(3, n), plv(3, 3), c(n, n), tau, lv(3, nli), kv(3, nkv)
        real(8), intent(inout) :: energy
        real(8), intent(inout), optional :: grad1(3, n), hess(3, n, 3, n), grad2(3, 3)
        interface
        subroutine rho_func(i, j, r2, mode, rho0, rho1, rho2)
            integer, intent(in) :: i, j, mode
            real(8), intent(in) :: r2
            real(8), intent(out) :: rho0, rho1, rho2
        end subroutine rho_func
        subroutine phi_func(k2, mode, phi0, phi1)
            integer, intent(in) :: mode
            real(8), intent(in) :: k2
            real(8), intent(out) :: phi0, phi1
        end subroutine phi_func
        end interface

        call ewald_add_real_sum(n, pos, nli, li, lv, rho_func, mode, energy, grad1, hess, grad2)
        call ewald_add_reciprocal_sum(n, pos, plv, nkv, kv, c, phi_func, tau, mode, energy, grad1, hess, grad2)
    end

    ! @ ==== dynamical matrix �̌v�Z ====
        
    ! dynamical matrix �̌v�Z�i����ԁA�P��Z���j
    subroutine ewald_add_dynamical_real_sum_in_single_cell(n, pos, rho_func, dyna)
        integer, intent(in) :: n
        real(8), intent(in) :: pos(3, n)
        complex(8), intent(inout) :: dyna(3, n, 3, n)
        interface
        subroutine rho_func(i, j, r2, mode, rho0, rho1, rho2)
            integer, intent(in) :: i, j, mode
            real(8), intent(in) :: r2
            real(8), intent(out) :: rho0, rho1, rho2
        end subroutine rho_func
        end interface

        integer :: i, j, s, t
        real(8) :: rij(3), rij2, rho0, rho1, rho2, h(3, 3)

        do i = 1, n
        do j = 1, i - 1
            rij(1:3) = pos(1:3, i) - pos(1:3, j)
            rij2 = sum(rij*rij)
            call rho_func(i, j, rij2, 2, rho0, rho1, rho2)

            do s = 1, 3
            do t = 1, 3
                h(s, t) = rho2*rij(s)*rij(t)
                if (s==t) then
                h(s, t) = h(s, t) + rho1
                end if
            end do
            end do
            dyna(1:3, i, 1:3, i) = dyna(1:3, i, 1:3, i) + h(1:3, 1:3)
            dyna(1:3, j, 1:3, j) = dyna(1:3, j, 1:3, j) + h(1:3, 1:3)
            dyna(1:3, i, 1:3, j) = dyna(1:3, i, 1:3, j) - h(1:3, 1:3)
            dyna(1:3, j, 1:3, i) = dyna(1:3, j, 1:3, i) - h(1:3, 1:3)

        end do
        end do
    end
        
    ! dynamical matrix �̌v�Z�i����ԁA�S�Z���j
    subroutine ewald_add_dynamical_real_sum(n, pos, nli, li, lv, rho_func, qv, dyna)
        integer, intent(in) :: n, nli, li(3, nli)
        real(8), intent(in) :: pos(3, n), lv(3, nli), qv(3)
        complex(8), intent(inout) :: dyna(3, n, 3, n)
        interface
        subroutine rho_func(i, j, r2, mode, rho0, rho1, rho2)
            integer, intent(in) :: i, j, mode
            real(8), intent(in) :: r2
            real(8), intent(out) :: rho0, rho1, rho2
        end subroutine rho_func
        end interface

        integer :: i, j, l, s, t
        real(8) :: rijl(3), rijl2, rho0, rho1, rho2, h(3, 3)
        complex(8), parameter :: i_const = (0d0, 1d0)

        !$omp parallel private (i, j, l, s, t, rijl, rijl2, rho0, rho1, rho2, h)
        !$omp do reduction (+:dyna)
        do l = 1, nli
        do i = 1, n
            do j = 1, n

            rijl(:) = pos(:,i) - pos(:,j) + lv(:,l)
            rijl2 = sum(rijl*rijl)
            call rho_func(i, j, rijl2, 2, rho0, rho1, rho2)

            do s = 1, 3
                do t = 1, 3
                h(s,t) = rho2 * rijl(s) * rijl(t)
                end do
            end do
            do s = 1, 3
                h(s,s) = h(s,s) + rho1
            end do
            
            ! �Ίp�����̌v�Z
            dyna(:,i,:,i) = dyna(:,i,:,i) + h(:,:)
            dyna(:,j,:,j) = dyna(:,j,:,j) + h(:,:)

            ! ��Ίp�����̌v�Z
            dyna(:,i,:,j) = dyna(:,i,:,j) - exp( dot_product(qv(:), lv(:,l)) * i_const) * h(:,:)
            dyna(:,j,:,i) = dyna(:,j,:,i) - exp(-dot_product(qv(:), lv(:,l)) * i_const) * h(:,:)

            end do
        end do
        end do
        !$omp end do
        !$omp end parallel

        call ewald_add_dynamical_real_sum_in_single_cell(n, pos, rho_func, dyna)

    end
    
    ! dynamical matrix �̌v�Z�i�t��ԁA�S�Z���j
    subroutine ewald_add_dynamical_reciprocal_sum(n, pos, plv, nkv, kv, c, phi_func, qv, dyna)
        integer, intent(in) :: n, nkv
        real(8), intent(in) :: pos(3, n), plv(3, 3), kv(3, nkv), c(n, n), qv(3)
        complex(8), intent(inout) :: dyna(3, n, 3, n)
        interface
        subroutine phi_func(k2, mode, phi0, phi1)
            integer, intent(in) :: mode
            real(8), intent(in) :: k2
            real(8), intent(out) :: phi0, phi1
        end subroutine phi_func
        end interface

        integer :: i, j, k, s, t
        real(8) :: volume, inv_volume
        real(8) :: phi0, phi1, hkv(3)
        real(8) :: pf_hess(3, 3)
        real(8) :: h(3, 3)
        real(8), parameter :: pi_times_2 = const_pi*2d0

        volume = abs(mat_determinant(plv))
        inv_volume = 1d0/volume

        ! non-zero k term
        !$omp parallel shared(n, nkv, pos, kv, qv, dyna, inv_volume) &
        !$omp private(i, j, k, s, t, phi0, phi1, hkv, pf_hess, h)
        !$omp do reduction(+:dyna)
        do k = 1, nkv
        
        ! �Ίp�����̌v�Z
        hkv(:) = kv(:,k)
        call phi_func(sum(hkv(:)*hkv(:)), 2, phi0, phi1)

        do s = 1, 3
            do t = 1, 3
            pf_hess(s, t) = -2*inv_volume*phi0*hkv(s)*hkv(t)
            end do
        end do

        do i = 1, n
            do j = 1, i - 1
            h(:, :) = c(i, j) * cos(dot_product(hkv(:), pos(:, i) - pos(:, j))) * pf_hess(:, :)
            dyna(:, i, :, i) = dyna(:, i, :, i) + h(:, :)
            dyna(:, j, :, j) = dyna(:, j, :, j) + h(:, :)
            end do
        end do

        !! ��Ίp�����̌v�Z
        !hkv(:) = kv(:,k) + qv(:)
    !   call phi_func(sum(hkv(:)*hkv(:)), 2, phi0, phi1)
        !
    !   do s = 1, 3
    !     do t = 1, 3
    !       pf_hess(s, t) = -2*inv_volume*phi0*hkv(s)*hkv(t)
    !     end do
    !   end do
    !
    !   do i = 1, n
    !     do j = 1, i - 1
    !       h(:, :) = c(i, j) * cos(dot_product(hkv(:), pos(:, i) - pos(:, j))) * pf_hess(:, :)
    !       dyna(:, i, :, j) = dyna(:, i, :, j) - h(:, :)
    !       dyna(:, j, :, i) = dyna(:, j, :, i) - h(:, :)
    !     end do
    !   end do

        ! ��Ίp�����̌v�Z�i���̂P�j
        hkv(:) = kv(:,k) + qv(:)
        call phi_func(sum(hkv(:)*hkv(:)), 2, phi0, phi1)
        
        do s = 1, 3
            do t = 1, 3
            pf_hess(s, t) = -inv_volume*phi0*hkv(s)*hkv(t)
            end do
        end do
    
        do i = 1, n
            do j = 1, i - 1
            h(:, :) = c(i, j) * cos(dot_product(hkv(:), pos(:, i) - pos(:, j))) * pf_hess(:, :)
            dyna(:, i, :, j) = dyna(:, i, :, j) - h(:, :)
            dyna(:, j, :, i) = dyna(:, j, :, i) - h(:, :)
            end do
        end do

        ! ��Ίp�����̌v�Z�i���̂Q�j
        hkv(:) = -kv(:,k) + qv(:)
        call phi_func(sum(hkv(:)*hkv(:)), 2, phi0, phi1)
        
        do s = 1, 3
            do t = 1, 3
            pf_hess(s, t) = -inv_volume*phi0*hkv(s)*hkv(t)
            end do
        end do
    
        do i = 1, n
            do j = 1, i - 1
            h(:, :) = c(i, j) * cos(dot_product(hkv(:), pos(:, i) - pos(:, j))) * pf_hess(:, :)
            dyna(:, i, :, j) = dyna(:, i, :, j) - h(:, :)
            dyna(:, j, :, i) = dyna(:, j, :, i) - h(:, :)
            end do
        end do

        !! ��Ίp�����̌v�Z�i���̂R�j
        !hkv(:) = qv(:)
    !   call phi_func(sum(hkv(:)*hkv(:)), 2, phi0, phi1)
        !
    !   do s = 1, 3
    !     do t = 1, 3
    !       pf_hess(s, t) = -inv_volume*phi0*hkv(s)*hkv(t)
    !     end do
    !   end do
    !
    !   do i = 1, n
    !     do j = 1, i - 1
    !       h(:, :) = c(i, j) * cos(dot_product(hkv(:), pos(:, i) - pos(:, j))) * pf_hess(:, :)
    !       dyna(:, i, :, j) = dyna(:, i, :, j) - h(:, :)
    !       dyna(:, j, :, i) = dyna(:, j, :, i) - h(:, :)
    !     end do
    !   end do
        
        end do
        !$omp end do
        !$omp end parallel

        ! kv = 0 �̍�
        
        ! �Ίp�����̌v�Z
        hkv(:) = 0d0
        call phi_func(sum(hkv(:)*hkv(:)), 2, phi0, phi1)

        do s = 1, 3
        do t = 1, 3
            pf_hess(s, t) = -2*inv_volume*phi0*hkv(s)*hkv(t)
        end do
        end do

        do i = 1, n
        do j = 1, i - 1
            h(:, :) = c(i, j) * cos(dot_product(hkv(:), pos(:, i) - pos(:, j))) * pf_hess(:, :)
            dyna(:, i, :, i) = dyna(:, i, :, i) + h(:, :)
            dyna(:, j, :, j) = dyna(:, j, :, j) + h(:, :)
        end do
        end do

        ! ��Ίp�����̌v�Z
        hkv(:) = qv(:)
        call phi_func(sum(hkv(:)*hkv(:)), 2, phi0, phi1)
        
        do s = 1, 3
        do t = 1, 3
            pf_hess(s, t) = -inv_volume*phi0*hkv(s)*hkv(t)
        end do
        end do
    
        do i = 1, n
        do j = 1, i - 1
            h(:, :) = c(i, j) * cos(dot_product(hkv(:), pos(:, i) - pos(:, j))) * pf_hess(:, :)
            dyna(:, i, :, j) = dyna(:, i, :, j) - h(:, :)
            dyna(:, j, :, i) = dyna(:, j, :, i) - h(:, :)
        end do
        end do

    end

    ! dynamical matrix �̌v�Z
    subroutine ewald_add_dynamical_ewald_sum(n, pos, plv, c, nli, li, lv, nkv, kv, rho_func, phi_func, qv, dyna)
        integer, intent(in) :: n, nli, li(3, nli), nkv
        real(8), intent(in) :: pos(3, n), plv(3, 3), c(n, n), lv(3, nli), kv(3, nkv), qv(3)
        complex(8), intent(inout) :: dyna(3, n, 3, n)
        interface
        subroutine rho_func(i, j, r2, mode, rho0, rho1, rho2)
            integer, intent(in) :: i, j, mode
            real(8), intent(in) :: r2
            real(8), intent(out) :: rho0, rho1, rho2
        end subroutine rho_func
        subroutine phi_func(k2, mode, phi0, phi1)
            integer, intent(in) :: mode
            real(8), intent(in) :: k2
            real(8), intent(out) :: phi0, phi1
        end subroutine phi_func
        end interface

        call ewald_add_dynamical_real_sum(n, pos, nli, li, lv, rho_func, qv, dyna)
        call ewald_add_dynamical_reciprocal_sum(n, pos, plv, nkv, kv, c, phi_func, qv, dyna)
    end


    ! 2020/09/12
    subroutine lsum_energy(n, pos, plv, nli, li, npar, par, efun, energy)
        integer, intent(in) :: n, nli, li(3, nli), npar
        real(8), intent(in) :: pos(3, n), plv(3, 3), par(npar, n, n)
        real(8), intent(inout) :: energy
        interface
            pure function efun(intercell, npar, par, rv)
                logical, intent(in) :: intercell
                integer, intent(in) :: npar
                real(8), intent(in) :: par(npar), rv(3)
                real(8) :: efun
            end
        end interface

        integer :: l, i, j
        real(8) :: lv(3)

        !$omp parallel do private(lv) reduction(+: energy)
        do l = 1, nli
            lv(:) = matmul(plv(:, :), li(:, l))
            do j = 1, n
                do i = 1, n
                    energy = energy + efun(.true., npar, par(:, i, j), pos(:, i) - pos(:, j) + lv(:))
                end do
            end do
        end do
        !$omp end parallel do

        !$omp parallel do reduction(+: energy) schedule(static, 1)
        do j = 1, n
            do i = 1, j - 1
                energy = energy + efun(.false., npar, par(:, i, j), pos(:, i) - pos(:, j))
            end do
        end do
        !$omp end parallel do
    end

    ! 2020/09/12
    subroutine lsum_gradients(n, pos, plv, nli, li, npar, par, gfun, grad1, grad2)
        integer, intent(in) :: n, nli, li(3, nli), npar
        real(8), intent(in) :: pos(3, n), plv(3, 3), par(npar, n, n)
        real(8), intent(inout) :: grad1(3, n), grad2(3, 3)
        interface
            pure function gfun(intercell, npar, par, rv)
                logical, intent(in) :: intercell
                integer, intent(in) :: npar
                real(8), intent(in) :: par(npar), rv(3)
                real(8) :: gfun(3)
            end
        end interface

        integer :: l, i, j
        real(8) :: lv(3), g(3)

        !$omp parallel do private(lv, g) reduction(+: grad1, grad2)
        do l = 1, nli
            lv(:) = matmul(plv(:, :), li(:, l))
            do j = 1, n
                do i = 1, n
                    g(:) = gfun(.true., npar, par(:, i, j), pos(:, i) - pos(:, j) + lv(:))
                    grad1(:, i) = grad1(:, i) + g(:)
                    grad1(:, j) = grad1(:, j) - g(:)
                    grad2(:, 1) = grad2(:, 1) + g(:) * li(1, l)
                    grad2(:, 2) = grad2(:, 2) + g(:) * li(2, l)
                    grad2(:, 3) = grad2(:, 3) + g(:) * li(3, l)
                end do
            end do
        end do
        !$omp end parallel do

        !$omp parallel do private(g) reduction(+: grad1) schedule(static, 1)
        do j = 1, n
            do i = 1, j - 1
                g(:) = gfun(.false., npar, par(:, i, j), pos(:, i) - pos(:, j))
                grad1(:, i) = grad1(:, i) + g(:)
                grad1(:, j) = grad1(:, j) - g(:)
            end do
        end do
        !$omp end parallel do
    end

    ! 2020/09/12
    subroutine lsum_hessian(n, pos, plv, nli, li, npar, par, hfun, hess)
        integer, intent(in) :: n, nli, li(3, nli), npar
        real(8), intent(in) :: pos(3, n), plv(3, 3), par(npar, n, n)
        real(8), intent(inout) :: hess(3, n, 3, n)
        interface
            pure function hfun(intercell, npar, par, rv)
                logical, intent(in) :: intercell
                integer, intent(in) :: npar
                real(8), intent(in) :: par(npar), rv(3)
                real(8) :: hfun(3, 3)
            end
        end interface

        integer :: l, i, j
        real(8) :: lv(3), h(3, 3)

        !$omp parallel do private(lv, h) reduction(+: hess)
        do l = 1, nli
            lv(:) = matmul(plv(:, :), li(:, l))
            do j = 1, n
                do i = 1, n
                    h(:, :) = hfun(.true., npar, par(:, i, j), pos(:, i) - pos(:, j) + lv(:))
                    hess(:, i, :, i) = hess(:, i, :, i) + h(:, :)
                    hess(:, j, :, j) = hess(:, j, :, j) + h(:, :)
                    hess(:, i, :, j) = hess(:, i, :, j) - h(:, :)
                    hess(:, j, :, i) = hess(:, j, :, i) - h(:, :)
                end do
            end do
        end do
        !$omp end parallel do

        !$omp parallel do private(h) reduction(+: hess) schedule(static, 1)
        do j = 1, n
            do i = 1, j - 1
                h(:, :) = hfun(.false., npar, par(:, i, j), pos(:, i) - pos(:, j))
                hess(:, i, :, i) = hess(:, i, :, i) + h(:, :)
                hess(:, j, :, j) = hess(:, j, :, j) + h(:, :)
                hess(:, i, :, j) = hess(:, i, :, j) - h(:, :)
                hess(:, j, :, i) = hess(:, j, :, i) - h(:, :)
            end do
        end do
        !$omp end parallel do
    end

    ! 2020/09/13
    subroutine lsum_dynamical(n, pos, plv, nli, li, npar, par, hfun, qv, dyna)
        integer, intent(in) :: n, nli, li(3, nli), npar
        real(8), intent(in) :: pos(3, n), plv(3, 3), par(npar, n, n), qv(3)
        real(8), intent(inout) :: dyna(2, 3, n, 3, n)
        interface
            pure function hfun(intercell, npar, par, rv)
                logical, intent(in) :: intercell
                integer, intent(in) :: npar
                real(8), intent(in) :: par(npar), rv(3)
                real(8) :: hfun(3, 3)
            end
        end interface

        integer :: l, i, j
        real(8) :: lv(3), h(3, 3)
        complex(8) :: phase, iphase

        !$omp parallel do private(lv, h, phase, iphase) reduction(+: dyna)
        do l = 1, nli
            lv(:) = matmul(plv(:, :), li(:, l))
            do j = 1, n
                do i = 1, n
                    h(:, :) = hfun(.true., npar, par(:, i, j), pos(:, i) - pos(:, j) + lv(:))
                    phase = exp(cmplx(0d0, dot_product(qv(:), lv(:)), kind(0d0)))
                    iphase = 1d0 / phase
                    dyna(1, :, i, :, i) = dyna(1, :, i, :, i) + h(:, :)
                    dyna(1, :, j, :, j) = dyna(1, :, j, :, j) + h(:, :)
                    dyna(1, :, i, :, j) = dyna(1, :, i, :, j) - h(:, :) * real(phase)
                    dyna(2, :, i, :, j) = dyna(2, :, i, :, j) - h(:, :) * aimag(phase)
                    dyna(1, :, j, :, i) = dyna(1, :, j, :, i) - h(:, :) * real(iphase)
                    dyna(2, :, j, :, i) = dyna(2, :, j, :, i) - h(:, :) * aimag(iphase)
                end do
            end do
        end do
        !$omp end parallel do

        !$omp parallel do private(h) reduction(+: dyna) schedule(static, 1)
        do j = 1, n
            do i = 1, j - 1
                h(:, :) = hfun(.false., npar, par(:, i, j), pos(:, i) - pos(:, j))
                dyna(1, :, i, :, i) = dyna(1, :, i, :, i) + h(:, :)
                dyna(1, :, j, :, j) = dyna(1, :, j, :, j) + h(:, :)
                dyna(1, :, i, :, j) = dyna(1, :, i, :, j) - h(:, :)
                dyna(1, :, j, :, i) = dyna(1, :, j, :, i) - h(:, :)
            end do
        end do
        !$omp end parallel do
    end

end module
