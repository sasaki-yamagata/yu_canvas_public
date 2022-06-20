module bonded_force_field

    use :: common, only: vec_angle, vec_cross_product
    implicit none

    contains

    real(8) pure function e_stretch(pos, par) result(e)
        real(8), intent(in) :: pos(:, :), par(:)
        e = 0.5d0 * par(2) * (norm2(pos(:, 2) - pos(:, 1)) - par(1))**2
    end function

    real(8) pure function e_bend(pos, par) result(e)
        real(8), intent(in) :: pos(:, :), par(:)
        real(8), parameter :: deg_to_rad = atan(1d0) / 45d0
        e = 0.5d0 * par(2) * (vec_angle(pos(:, 1) - pos(:, 2), pos(:, 3) - pos(:, 2)) - par(1) * deg_to_rad)**2
    end function

    real(8) pure function e_bend_cos(pos, par) result(e)
        real(8), intent(in) :: pos(:, :), par(:)
        real(8), parameter :: deg_to_rad = atan(1d0) / 45d0
        real(8) :: a(3), b(3)
        a(:) = pos(:, 1) - pos(:, 2)
        b(:) = pos(:, 3) - pos(:, 2)
        e = 0.5d0 * par(2) * (dot_product(a,b)/sqrt(sum(a*a)*sum(b*b)) - cos(par(1) * deg_to_rad))**2
    end function

    real(8) pure function e_torsion(pos, par) result(e)
        real(8), intent(in) :: pos(:, :), par(:)
        real(8) :: theta, v(3, 3)
        v(:, 1) = pos(:, 2) - pos(:, 1)
        v(:, 2) = pos(:, 3) - pos(:, 2)
        v(:, 3) = pos(:, 4) - pos(:, 3)
        theta = vec_angle(vec_cross_product(v(:, 1), v(:, 2)), vec_cross_product(v(:, 2), v(:, 3)))
        e = 0.5d0 * (par(1) * (1d0 - cos(theta)) + par(2) * (1d0 - cos(theta+theta)) + par(3) * (1d0 - cos(theta+theta+theta)))
    end function

    real(8) pure function e_inversion(pos, par) result(e)
        real(8), intent(in) :: pos(:, :), par(:)
        real(8) :: theta, v(3, 3)
        real(8), parameter :: right_angle = 2 * atan(1d0)
        v(:, 1) = pos(:, 2) - pos(:, 1)
        v(:, 2) = pos(:, 3) - pos(:, 1)
        v(:, 3) = pos(:, 4) - pos(:, 1)
        theta = vec_angle(vec_cross_product(v(:, 1), v(:, 2)), v(:, 3)) - right_angle
        e = 0.5d0 * par(1) * theta * theta
    end function

    real(8) pure function e_inversion_cos_planar(pos, par) result(e)
        real(8), intent(in) :: pos(:, :), par(:)
        real(8) :: a(3), b(3), cos_psi
        real(8), parameter :: inv_3 = 1d0 / 3d0
        a(:) = pos(:, 2) - pos(:, 1)
        b(:) = vec_cross_product(pos(:, 3) - pos(:, 1), pos(:, 4) - pos(:, 1))
        cos_psi = dot_product(a,b)/sqrt(sum(a*a)*sum(b*b))
        e = par(1) * (1.0 - sqrt(max(0d0, 1.0 - cos_psi * cos_psi)))

        a(:) = pos(:, 3) - pos(:, 1)
        b(:) = vec_cross_product(pos(:, 4) - pos(:, 1), pos(:, 2) - pos(:, 1))
        cos_psi = dot_product(a,b)/sqrt(sum(a*a)*sum(b*b))
        e = e + par(1) * (1.0 - sqrt(max(0d0, 1.0 - cos_psi * cos_psi)))

        a(:) = pos(:, 4) - pos(:, 1)
        b(:) = vec_cross_product(pos(:, 2) - pos(:, 1), pos(:, 3) - pos(:, 1))
        cos_psi = dot_product(a,b)/sqrt(sum(a*a)*sum(b*b))
        e = e + par(1) * (1.0 - sqrt(max(0d0, 1.0 - cos_psi * cos_psi)))

        e = e * inv_3
    end function

    !---- add new force field here (1/2) ----!

    function e_func_from_name(nname, name) result(func)
        integer, intent(in) :: nname
        character, intent(in) :: name(nname)
        procedure(e_stretch), pointer :: func
        character(nname) :: name2
        integer :: i

        do i = 1, nname
            name2(i:i) = name(i)
        end do

        ! select force field type
        select case(name2)
        case("stretch")
            func => e_stretch
        case("bend")
            func => e_bend
        case("bend_cos")
            func => e_bend_cos
        case("torsion")
            func => e_torsion
        case("inversion")
            func => e_inversion
        case("inversion_cos_planar")
            func => e_inversion_cos_planar

        !---- add new force field here (2/2) ----!

        case default
            print *, "Force field type not found: ", name
            stop
        end select

    end function

    pure function g_from_e(func, na, pos, par, delta) result(g)
        procedure(e_stretch), pointer :: func
        integer, intent(in) :: na
        real(8), intent(in) :: pos(3, na), par(:), delta
        real(8) :: g(3, na)
        
        integer :: s, i
        real(8) :: a(3, na)

        g(:, :) = 0d0
        a(:, :) = pos(:, :)

        do s = 1, 3
            do i = 2, na
                a(s, i) = pos(s, i) + delta
                g(s, i) = func(a, par)
                a(s, i) = pos(s, i) - delta
                g(s, i) = (g(s, i) - func(a, par)) / (delta + delta)
                a(s, i) = pos(s, i)
            end do
        end do
        
        forall(s=1:3) g(s, 1) = -sum(g(s, 2:na))
    end function

    pure function h_from_e(func, na, pos, par, delta) result(h)
        procedure(e_stretch), pointer :: func
        integer, intent(in) :: na
        real(8), intent(in) :: pos(3, na), par(:), delta
        real(8) :: h(3, na, 3, na)
        
        integer :: i, j, s, t
        real(8) :: a(3, na)

        h(:, :, :, :) = 0d0
        a(:, :) = pos(:, :)

        ! off-diagonal
        do j = 2, na
            do i = 2, na
                do t = 1, 3
                    do s = 1, 3
                        if  (i<j .or. (i==j .and. s<t)) then
                            a(s, i) = pos(s, i) + delta
                            a(t, j) = pos(t, j) + delta
                            h(s, i, t, j) = func(a, par)
                            a(s, i) = pos(s, i) + delta
                            a(t, j) = pos(t, j) - delta
                            h(s, i, t, j) = h(s, i, t, j) - func(a, par)
                            a(s, i) = pos(s, i) - delta
                            a(t, j) = pos(t, j) + delta
                            h(s, i, t, j) = h(s, i, t, j) - func(a, par)
                            a(s, i) = pos(s, i) - delta
                            a(t, j) = pos(t, j) - delta
                            h(s, i, t, j) = h(s, i, t, j) + func(a, par)
                            a(s, i) = pos(s, i)
                            a(t, j) = pos(t, j)
                            h(s, i, t, j) = h(s, i, t, j) / (delta + delta)**2
                            h(t, j, s, i) = h(s, i, t, j)
                        end if
                    end do
                end do
            end do
        end do

        ! diagonal
        do i = 2, na
            do s = 1, 3
                h(s, i, s, i) = -2 * func(a, par)
                a(s, i) = pos(s, i) + delta
                h(s, i, s, i) = h(s, i, s, i) + func(a, par)
                a(s, i) = pos(s, i) - delta
                h(s, i, s, i) = h(s, i, s, i) + func(a, par)
                a(s, i) = pos(s, i)
                h(s, i, s, i) = h(s, i, s, i) / delta**2
            end do
        end do

        ! first atom (i == 1 .or. j == 1)
        forall(s=1:3, i=2:na, t=1:3) h(s, i, t, 1) = -sum(h(s, i, t, 2:na))
        forall(s=1:3, t=1:3, j=1:na) h(s, 1, t, j) = -sum(h(s, 2:na, t, j))
    end function

end module

module non_bonded_force_field

    implicit none

    contains

    real(8) pure function rho_lj_6_12(r2, par) result(rho)
        real(8), intent(in) :: r2, par(:)
        real(8) :: inv_r6
        inv_r6 = 1d0 / (r2 * r2 * r2)
        rho = (par(1) * inv_r6 + par(2)) * inv_r6
    end function

    real(8) pure function rho_lj_1_6_12(r2, par) result(rho)
        real(8), intent(in) :: r2, par(:)
        real(8) :: inv_r6
        inv_r6 = 1d0 / (r2 * r2 * r2)
        rho = (par(1) * inv_r6 + par(2)) * inv_r6 + par(3) / sqrt(r2)
    end function

    real(8) pure function rho_buck(r2, par) result(rho)
        real(8), intent(in) :: r2, par(:)
        rho = par(1) * exp(-par(2) * sqrt(r2)) + par(3) / (r2 * r2 * r2)
    end function

    real(8) pure function rho_coul_buck(r2, par) result(rho)
        real(8), intent(in) :: r2, par(:)
        rho = par(1) * exp(-par(2) * sqrt(r2)) + par(3) / (r2 * r2 * r2) + par(4) / sqrt(r2)
    end function

    ! real(8) pure function e_lj_6_12_ewald(pos, par) result(e)
    !     integer, intent(in) :: na, npar
    !     real(8), intent(in) :: pos(3, na), par(npar)
    !     real(8) :: r2, r6, d2, a2, a4
    !     r2 = sum((pos(:, 2) - pos(:, 1))**2)
    !     r6 = r2**3
    !     d2 = par(3)**2
    !     a2 = r2 / d2
    !     a4 = a2 * a2
    !     e = par(1) / (r6 * r6) + par(2) * (1d0 + a2 + 0.5d0 * a4) * exp(a2) / r6
    ! end

    ! real(8) pure function e_buckingham_ewald(pos, par) result(e)
    !     integer, intent(in) :: na, npar
    !     real(8), intent(in) :: pos(3, na), par(npar)
    !     real(8) :: r2, r6, d2, a2, a4
    !     r2 = sum((pos(:, 2) - pos(:, 1))**2)
    !     r6 = r2**3
    !     d2 = par(4)**2
    !     a2 = r2 / d2
    !     a4 = a2 * a2
    !     e = par(1) * exp(-par(2) * sqrt(r2)) + par(3) * (1d0 + a2 + 0.5d0 * a4) * exp(a2) / r6
    ! end

    !---- add new force field here (1/2) ----!

    function rho_func_from_name(nname, name) result(func)
        integer, intent(in) :: nname
        character, intent(in) :: name(nname)
        procedure(rho_lj_6_12), pointer :: func

        character(nname) :: name2
        integer :: i

        do i = 1, nname
            name2(i:i) = name(i)
        end do

        ! select force field type
        select case(name2)
        case("lj_6_12")
            func => rho_lj_6_12
        case("lj_1_6_12")
            func => rho_lj_1_6_12
        case("buck")
            func => rho_buck
        case("coul_buck")
            func => rho_coul_buck

        !---- add new force field here (2/2) ----!

        case default
            print *, "Force field type not found: ", name
            stop
        end select

    end function

    real(8) pure function e_from_rho(func, rv, par) result(e)
        procedure(rho_lj_6_12), pointer :: func
        real(8), intent(in) :: rv(3), par(:)
        e = func(sum(rv(:)**2), par)
    end function

    pure function g_from_rho(func, rv, par, delta2) result(g)
        procedure(rho_lj_6_12), pointer :: func
        real(8), intent(in) :: rv(3), par(:), delta2
        real(8) :: g(3)
        real(8) :: r2, e(2), delta3
        r2 = sum(rv(:)**2)
        delta3 = delta2 * r2
        e(1) = func(r2 - delta3, par)
        e(2) = func(r2 + delta3, par)
        g(:) = 2 * (e(2) - e(1)) / (delta3 + delta3) * rv(:)
    end function

    pure function h_from_rho(func, rv, par, delta2) result(h)
        procedure(rho_lj_6_12), pointer :: func
        real(8), intent(in) :: rv(3), par(:), delta2
        real(8) :: h(3, 3)
        real(8) :: r2, e(3), rho1, rho2, delta3
        integer :: s, t
        r2 = sum(rv(:)**2)
        delta3 = delta2 * r2
        e(1) = func(r2 - delta3, par)
        e(2) = func(r2, par)
        e(3) = func(r2 + delta3, par)
        rho1 = (e(3) - e(1)) / (delta3 + delta3)
        rho2 = (e(3) - e(2) - e(2) + e(1)) / (delta3 * delta3)
        forall(s=1:3, t=1:3) h(s, t) = 4 * rho2 * rv(s) * rv(t)
        forall(s=1:3) h(s, s) = h(s, s) + 2 * rho1
    end function

end module

!returns bonded force field energy, called by Python
function bonded_energy(nname, name, na, nmol, pos, nb, nidx, idx, npar, par) result(e) bind(c)
    !DEC$ ATTRIBUTES DLLEXPORT :: bonded_energy
    use :: bonded_force_field
    use :: iso_c_binding
    implicit none
    integer(c_int), intent(in) :: na, nmol, nb, nidx, npar, nname, idx(nidx, nb)
    real(c_double), intent(in) :: pos(3, na, nmol), par(npar, nb)
    character(c_char), intent(in) :: name(nname)
    real(c_double) :: e

    procedure(e_stretch), pointer :: func
    integer :: i, j

    func => e_func_from_name(nname, name)

    e = 0d0
    !$omp parallel do reduction(+:e)
    do i = 1, nb
        do j = 1, nmol
            e = e + func(pos(:, idx(:, i), j), par(:, i))
        end do
    end do
    !$omp end parallel do

end

!accumulate bonded force field gradient, called by Python
subroutine bonded_gradient(nname, name, na, nmol, pos, nb, nidx, idx, npar, par, delta, grad) bind(c)
    !DEC$ ATTRIBUTES DLLEXPORT :: bonded_gradient
    use :: bonded_force_field
    use :: iso_c_binding
    implicit none
    integer(c_int), intent(in) :: na, nmol, nb, nidx, npar, nname, idx(nidx, nb)
    real(c_double), intent(in) :: pos(3, na, nmol), par(npar, nb), delta
    character(c_char), intent(in) :: name(nname)
    real(c_double), intent(inout) :: grad(3, na, nmol)

    procedure(e_stretch), pointer :: func
    integer :: i, j

    func => e_func_from_name(nname, name)

    !$omp parallel do reduction(+:grad)
    do i = 1, nb
        do j = 1, nmol
            grad(:, idx(:, i), j) = grad(:, idx(:, i), j) + &
                g_from_e(func, nidx, pos(:, idx(:, i), j), par(:, i), delta)
        end do
    end do
    !$omp end parallel do

end

!accumulate bonded force field hessian, called by Python
subroutine bonded_hessian(nname, name, na, nmol, pos, nb, nidx, idx, npar, par, delta, hess) bind(c)
    !DEC$ ATTRIBUTES DLLEXPORT :: bonded_hessian
    use :: bonded_force_field
    use :: iso_c_binding
    implicit none
    integer(c_int), intent(in) :: na, nmol, nb, nidx, npar, nname, idx(nidx, nb)
    real(c_double), intent(in) :: pos(3, na, nmol), par(npar, nb), delta
    character(c_char), intent(in) :: name(nname)
    real(c_double), intent(inout) :: hess(3, na, nmol, 3, na, nmol)

    procedure(e_stretch), pointer :: func
    integer :: i, j

    func => e_func_from_name(nname, name)

    !$omp parallel do reduction(+:hess)
    do i = 1, nb
        do j = 1, nmol
            hess(:, idx(:, i), j, :, idx(:, i), j) = hess(:, idx(:, i), j, :, idx(:, i), j) + &
                h_from_e(func, nidx, pos(:, idx(:, i), j), par(:, i), delta)
        end do
    end do
    !$omp end parallel do

end

!returns non-bonded force field energy, called by Python
function non_bonded_energy(nname, name, na, nmol, pos, plv, nli, li, npar, par, nglo, glo, exc) result(e) bind(c)
    !DEC$ ATTRIBUTES DLLEXPORT :: non_bonded_energy
    use :: non_bonded_force_field
    use :: iso_c_binding
    implicit none
    integer(c_int), intent(in) :: na, nmol, nli, npar, nglo, nname, li(3, nli)
    real(c_double), intent(in) :: pos(3, na, nmol), plv(3, 3), par(npar, na, na), glo(nglo), exc(na, na)
    character(c_char), intent(in) :: name(nname)
    real(c_double) :: e

    procedure(rho_lj_6_12), pointer :: func
    integer :: i, j, k, s, t
    real(8) :: lv(3), parglo(npar+nglo)

    func => rho_func_from_name(nname, name)
    e = 0d0
    parglo(npar+1:npar+nglo) = glo(:)

    !$omp parallel do private(lv, parglo) reduction(+: e)
    do k = 1, nli
        lv(:) = matmul(plv(:, :), li(:, k))
        do t = 1, nmol
            do s = 1, nmol
                do j = 1, na
                    do i = 1, na
                        parglo(1:npar) = par(:, i, j)
                        e = e + e_from_rho(func, pos(:, i, s) - pos(:, j, t) - lv(:), parglo)
                    end do
                end do
            end do
        end do
    end do
    !$omp end parallel do

    do t = 1, nmol
        do s = 1, t
            do j = 1, na
                if (s == t) then
                    do i = 1, j-1
                        parglo(1:npar) = par(:, i, j)
                        e = e + e_from_rho(func, pos(:, i, s) - pos(:, j, t), parglo) * (1d0 - exc(i, j))
                    end do
                else
                    do i = 1, na
                        parglo(1:npar) = par(:, i, j)
                        e = e + e_from_rho(func, pos(:, i, s) - pos(:, j, t), parglo)
                    end do
                end if
            end do
        end do
    end do

end

!accumulate non-bonded force field gradient, called by Python
subroutine non_bonded_gradient(nname, name, na, nmol, pos, plv, nli, li, npar, par, nglo, glo, exc, delta2, grad, grad2) bind(c)
    !DEC$ ATTRIBUTES DLLEXPORT :: non_bonded_gradient
    use :: non_bonded_force_field
    use :: iso_c_binding
    implicit none
    integer(c_int), intent(in) :: na, nmol, nli, npar, nglo, nname, li(3, nli)
    real(c_double), intent(in) :: pos(3, na, nmol), plv(3, 3), par(npar, na, na), glo(nglo), delta2, exc(na, na)
    character(c_char), intent(in) :: name(nname)
    real(c_double), intent(inout) :: grad(3, na, nmol), grad2(3, 3)

    procedure(rho_lj_6_12), pointer :: func
    integer :: i, j, k, s, t, u
    real(8) :: lv(3), parglo(npar+nglo), g(3)

    func => rho_func_from_name(nname, name)
    parglo(npar+1:npar+nglo) = glo(:)

    !$omp parallel do private(lv, parglo, g) reduction(+: grad, grad2)
    do k = 1, nli
        lv(:) = matmul(plv(:, :), li(:, k))
        do t = 1, nmol
            do s = 1, nmol
                do j = 1, na
                    do i = 1, na
                        parglo(1:npar) = par(:, i, j)
                        g(:) = g_from_rho(func, pos(:, i, s) - pos(:, j, t) - lv(:), parglo, delta2)
                        grad(:, i, s) = grad(:, i, s) + g(:)
                        grad(:, j, t) = grad(:, j, t) - g(:)
                        forall(u=1:3) grad2(:, u) = grad2(:, u) - g(:) * li(u, k)
                    end do
                end do
            end do
        end do
    end do
    !$omp end parallel do

    do t = 1, nmol
        do s = 1, t
            do j = 1, na
                if (s == t) then
                    do i = 1, j-1
                        parglo(1:npar) = par(:, i, j)
                        g(:) = g_from_rho(func, pos(:, i, s) - pos(:, j, t), parglo, delta2) * (1d0 - exc(i, j))
                        grad(:, i, s) = grad(:, i, s) + g(:)
                        grad(:, j, t) = grad(:, j, t) - g(:)
                    end do
                else
                    do i = 1, na
                        parglo(1:npar) = par(:, i, j)
                        g(:) = g_from_rho(func, pos(:, i, s) - pos(:, j, t), parglo, delta2)
                        grad(:, i, s) = grad(:, i, s) + g(:)
                        grad(:, j, t) = grad(:, j, t) - g(:)
                    end do
                end if
            end do
        end do
    end do
end

!accumulate non-bonded force field hessian, called by Python
subroutine non_bonded_hessian(nname, name, na, nmol, pos, plv, nli, li, npar, par, nglo, glo, exc, delta2, hess) bind(c)
    !DEC$ ATTRIBUTES DLLEXPORT :: non_bonded_hessian
    use :: non_bonded_force_field
    use :: iso_c_binding
    implicit none
    integer(c_int), intent(in) :: na, nmol, nli, npar, nglo, nname, li(3, nli)
    real(c_double), intent(in) :: pos(3, na, nmol), plv(3, 3), par(npar, na, na), glo(nglo), delta2, exc(na, na)
    character(c_char), intent(in) :: name(nname)
    real(c_double), intent(inout) :: hess(3, na, nmol, 3, na, nmol)

    procedure(rho_lj_6_12), pointer :: func
    integer :: i, j, k, s, t
    real(8) :: lv(3), parglo(npar+nglo), h(3, 3)

    func => rho_func_from_name(nname, name)
    parglo(npar+1:npar+nglo) = glo(:)

    !$omp parallel do private(lv, parglo, h) reduction(+: hess)
    do k = 1, nli
        lv(:) = matmul(plv(:, :), li(:, k))
        do t = 1, nmol
            do s = 1, nmol
                do j = 1, na
                    do i = 1, na
                        parglo(1:npar) = par(:, i, j)
                        h(:, :) = h_from_rho(func, pos(:, i, s) - pos(:, j, t) - lv(:), parglo, delta2)
                        hess(:, i, s, :, i, s) = hess(:, i, s, :, i, s) + h(:, :)
                        hess(:, j, t, :, i, s) = hess(:, j, t, :, i, s) - h(:, :)
                        hess(:, i, s, :, j, t) = hess(:, i, s, :, j, t) - h(:, :)
                        hess(:, j, t, :, j, t) = hess(:, j, t, :, j, t) + h(:, :)
                    end do
                end do
            end do
        end do
    end do
    !$omp end parallel do

    do t = 1, nmol
        do s = 1, t
            do j = 1, na
                if (s == t) then
                    do i = 1, j-1
                        parglo(1:npar) = par(:, i, j)
                        h(:, :) = h_from_rho(func, pos(:, i, s) - pos(:, j, t), parglo, delta2) * (1d0 - exc(i, j))
                        hess(:, i, s, :, i, s) = hess(:, i, s, :, i, s) + h(:, :)
                        hess(:, j, t, :, i, s) = hess(:, j, t, :, i, s) - h(:, :)
                        hess(:, i, s, :, j, t) = hess(:, i, s, :, j, t) - h(:, :)
                        hess(:, j, t, :, j, t) = hess(:, j, t, :, j, t) + h(:, :)
                    end do
                else
                    do i = 1, na
                        parglo(1:npar) = par(:, i, j)
                        h(:, :) = h_from_rho(func, pos(:, i, s) - pos(:, j, t), parglo, delta2)
                        hess(:, i, s, :, i, s) = hess(:, i, s, :, i, s) + h(:, :)
                        hess(:, j, t, :, i, s) = hess(:, j, t, :, i, s) - h(:, :)
                        hess(:, i, s, :, j, t) = hess(:, i, s, :, j, t) - h(:, :)
                        hess(:, j, t, :, j, t) = hess(:, j, t, :, j, t) + h(:, :)
                    end do
                end if
            end do
        end do
    end do
end

!accumulate non-bonded force field hessian, called by Python
subroutine non_bonded_hessians(nname, name, na, nmol, pos, plv, nli, li, npar, par, nglo, glo, &
    exc, delta2, hess11, hess12, hess22) bind(c)
    !DEC$ ATTRIBUTES DLLEXPORT :: non_bonded_hessians
    use :: non_bonded_force_field
    use :: iso_c_binding
    implicit none
    integer(c_int), intent(in) :: na, nmol, nli, npar, nglo, nname, li(3, nli)
    real(c_double), intent(in) :: pos(3, na, nmol), plv(3, 3), par(npar, na, na), glo(nglo), delta2, exc(na, na)
    character(c_char), intent(in) :: name(nname)
    real(c_double), intent(inout) :: hess11(3, na, nmol, 3, na, nmol), hess12(3, 3, 3, na, nmol), hess22(3, 3, 3, 3)

    procedure(rho_lj_6_12), pointer :: func
    integer :: i, j, k, s, t, u, v
    real(8) :: lv(3), parglo(npar+nglo), h(3, 3)

    func => rho_func_from_name(nname, name)
    parglo(npar+1:npar+nglo) = glo(:)

    !$omp parallel do private(lv, parglo, h) reduction(+: hess11, hess12, hess22)
    do k = 1, nli
        lv(:) = matmul(plv(:, :), li(:, k))
        do t = 1, nmol
            do s = 1, nmol
                do j = 1, na
                    do i = 1, na
                        parglo(1:npar) = par(:, i, j)
                        h(:, :) = h_from_rho(func, pos(:, i, s) - pos(:, j, t) - lv(:), parglo, delta2)
                        hess11(:, i, s, :, i, s) = hess11(:, i, s, :, i, s) + h(:, :)
                        hess11(:, j, t, :, i, s) = hess11(:, j, t, :, i, s) - h(:, :)
                        hess11(:, i, s, :, j, t) = hess11(:, i, s, :, j, t) - h(:, :)
                        hess11(:, j, t, :, j, t) = hess11(:, j, t, :, j, t) + h(:, :)
                        forall(u=1:3) hess12(:, u, :, i, s) = hess12(:, u, :, i, s) - h(:, :) * li(u, k)
                        forall(u=1:3) hess12(:, u, :, j, t) = hess12(:, u, :, j, t) + h(:, :) * li(u, k)
                        forall(u=1:3,v=1:3) hess22(:, u, :, v) = hess22(:, u, :, v) + h(:, :) * li(u, k) * li(v, k)
                    end do
                end do
            end do
        end do
    end do
    !$omp end parallel do

    do t = 1, nmol
        do s = 1, t
            do j = 1, na
                if (s == t) then
                    do i = 1, j-1
                        parglo(1:npar) = par(:, i, j)
                        h(:, :) = h_from_rho(func, pos(:, i, s) - pos(:, j, t), parglo, delta2) * (1d0 - exc(i, j))
                        hess11(:, i, s, :, i, s) = hess11(:, i, s, :, i, s) + h(:, :)
                        hess11(:, j, t, :, i, s) = hess11(:, j, t, :, i, s) - h(:, :)
                        hess11(:, i, s, :, j, t) = hess11(:, i, s, :, j, t) - h(:, :)
                        hess11(:, j, t, :, j, t) = hess11(:, j, t, :, j, t) + h(:, :)
                    end do
                else
                    do i = 1, na
                        parglo(1:npar) = par(:, i, j)
                        h(:, :) = h_from_rho(func, pos(:, i, s) - pos(:, j, t), parglo, delta2)
                        hess11(:, i, s, :, i, s) = hess11(:, i, s, :, i, s) + h(:, :)
                        hess11(:, j, t, :, i, s) = hess11(:, j, t, :, i, s) - h(:, :)
                        hess11(:, i, s, :, j, t) = hess11(:, i, s, :, j, t) - h(:, :)
                        hess11(:, j, t, :, j, t) = hess11(:, j, t, :, j, t) + h(:, :)
                    end do
                end if
            end do
        end do
    end do
end

!accumulate non-bonded force field dynamical, called by Python
subroutine non_bonded_dynamical(nname, name, na, pos, plv, nli, li, npar, par, nglo, glo, exc, delta2, qv, dyna) bind(c)
    !DEC$ ATTRIBUTES DLLEXPORT :: non_bonded_dynamical
    use :: non_bonded_force_field
    use :: iso_c_binding
    implicit none
    integer(c_int), intent(in) :: na, nli, npar, nglo, nname, li(3, nli)
    real(c_double), intent(in) :: pos(3, na), plv(3, 3), par(npar, na, na), glo(nglo), delta2, exc(na, na), qv(3)
    character(c_char), intent(in) :: name(nname)
    complex(c_double), intent(inout) :: dyna(3, na, 3, na)

    procedure(rho_lj_6_12), pointer :: func
    integer :: i, j, k
    real(8) :: lv(3), parglo(npar+nglo), h(3, 3)
    complex(8), parameter :: i_const = (0d0, 1d0)
    complex(8) :: exp_dpi(2)

    func => rho_func_from_name(nname, name)
    parglo(npar+1:npar+nglo) = glo(:)

    !$omp parallel do private(lv, parglo, h) reduction(+: dyna)
    do k = 1, nli
        lv(:) = matmul(plv(:, :), li(:, k))
        exp_dpi(1) = exp( dot_product(qv(:), lv(:)) * i_const)
        exp_dpi(2) = exp(-dot_product(qv(:), lv(:)) * i_const)
        do j = 1, na
            do i = 1, na
                parglo(1:npar) = par(:, i, j)
                h(:, :) = h_from_rho(func, pos(:, i) - pos(:, j) - lv(:), parglo, delta2)
                dyna(:, i, :, i) = dyna(:, i, :, i) + h(:, :)
                dyna(:, j, :, i) = dyna(:, j, :, i) + h(:, :) * exp_dpi(1)
                dyna(:, i, :, j) = dyna(:, i, :, j) + h(:, :) * exp_dpi(2)
                dyna(:, j, :, j) = dyna(:, j, :, j) + h(:, :)
            end do
        end do
    end do
    !$omp end parallel do

    do j = 1, na
        do i = 1, j - 1
            parglo(1:npar) = par(:, i, j)
            h(:, :) = h_from_rho(func, pos(:, i) - pos(:, j), parglo, delta2) * (1d0 - exc(i, j))
            dyna(:, i, :, i) = dyna(:, i, :, i) + h(:, :)
            dyna(:, j, :, i) = dyna(:, j, :, i) + h(:, :)
            dyna(:, i, :, j) = dyna(:, i, :, j) + h(:, :)
            dyna(:, j, :, j) = dyna(:, j, :, j) + h(:, :)
        end do
    end do
end


!returns ewald energy, called by Python
function ewald_energy(pow, na, nmol, pos, plv, nli, li, nki, ki, par, damp, exc) result(e) bind(c)
    !DEC$ ATTRIBUTES DLLEXPORT :: ewald_energy
    use :: common
    use :: iso_c_binding
    implicit none
    integer(c_int), intent(in) :: pow, na, nmol, nli, nki, li(3, nli), ki(3, nki)
    real(c_double), intent(in) :: pos(3, na, nmol), plv(3, 3), par(na, na), damp, exc(na, na)
    real(c_double) :: e

    integer :: i, j
    real(c_double) :: c(na * nmol, na * nmol), pkv(3, 3), lv(3, nli), kv(3, nki), cancel(na * nmol, na * nmol)
    real(c_double) :: d2, inv_d2, pf_rho0, pf_rho1, pf_rho2, pf_phi0, pf_phi1
    real(c_double), parameter :: pi_pow_3_2 = (4 * atan(1d0)) ** (1.5d0)

    forall(i=1:nmol, j=1:nmol) c((i-1)*nmol+1:(i-1)*nmol+na, (j-1)*nmol+1:(j-1)*nmol+na) = par(:, :)
    forall(i=1:nmol, j=1:nmol) cancel((i-1)*nmol+1:(i-1)*nmol+na, (j-1)*nmol+1:(j-1)*nmol+na) = exc(:, :)
    call ltc_reciprocal(plv, pkv)
    lv(:, :) = matmul(plv(:, :), li(:, :))
    kv(:, :) = matmul(pkv(:, :), ki(:, :))

    d2 = damp**2
    inv_d2 = 1d0 / damp**2
    pf_rho0 = 1d0 / damp**6
    pf_rho1 = 1d0 / damp**8
    pf_rho2 = 1d0 / damp**10
    pf_phi0 = 2*pi_pow_3_2/(3*damp**3)
    pf_phi1 = pi_pow_3_2/(2*damp)

    e = 0d0
    select case(pow)
    case(6)
        call ewald_add_ewald_sum(na * nmol, pos, plv, c, nli, li, lv, nki, kv, &
            rho_func_ewald_6, phi_func_ewald_6, 1d0/(12*damp**6), 0, e)
        call ewald_add_real_sum_in_single_cell(na * nmol, pos, rho_func_cancel_6, 0, e)
    end select
contains
    subroutine rho_func_ewald_6(i, j, r2, mode, rho0, rho1, rho2)
        integer, intent(in) :: i, j, mode
        real(8), intent(in) :: r2
        real(8), intent(out) :: rho0, rho1, rho2
        real(8) :: a2, a4, a6, a8, inv_a8, inv_a10, exp_a2
        a2 = r2*inv_d2
        a4 = a2*a2
        a6 = a4*a2
        exp_a2 = exp(-a2)
        select case (mode)
        case (0)
            rho0 = pf_rho0*c(i, j)*(1d0+a2+0.5d0*a4)*exp_a2/a6
        case (1, -1)
            inv_a8 = 1d0/(a4*a4)
            rho0 = pf_rho0*c(i, j)*(1d0+a2+0.5d0*a4)*exp_a2*a2*inv_a8
            rho1 = -pf_rho1*c(i, j)*(6d0+6*a2+3*a4+a6)*exp_a2*inv_a8
        case default ! 2 or -2
            a8 = a4*a4
            inv_a10 = 1d0/(a8*a2)
            rho0 = pf_rho0*c(i, j)*(1d0+a2+0.5d0*a4)*exp_a2*a4*inv_a10
            rho1 = -pf_rho1*c(i, j)*(6d0+6*a2+3*a4+a6)*exp_a2*a2*inv_a10
            rho2 = pf_rho2*c(i, j)*(48d0+48*a2+24*a4+8*a6+2*a8)*exp_a2*inv_a10
        end select
    end
    subroutine rho_func_cancel_6(i, j, r2, mode, rho0, rho1, rho2)
        integer, intent(in) :: i, j, mode
        real(8), intent(in) :: r2
        real(8), intent(out) :: rho0, rho1, rho2
        real(8) :: a2, a4, a6, a8, inv_a8, inv_a10
        a2 = r2*inv_d2
        a4 = a2*a2
        a6 = a4*a2
        select case (mode)
        case (0)
            rho0 = -pf_rho0*c(i, j)/a6*cancel(i, j)
        case (1, -1)
            inv_a8 = 1d0/(a4*a4)
            rho0 = -pf_rho0*c(i, j)*a2*inv_a8*cancel(i, j)
            rho1 = 6*pf_rho1*c(i, j)*inv_a8*cancel(i, j)
        case default ! 2 or -2
            a8 = a4*a4
            inv_a10 = 1d0/(a8*a2)
            rho0 = -pf_rho0*c(i, j)*a4*inv_a10*cancel(i, j)
            rho1 = 6*pf_rho1*c(i, j)*a2*inv_a10*cancel(i, j)
            rho2 = -48*pf_rho2*c(i, j)*inv_a10*cancel(i, j)
        end select
    end
    subroutine phi_func_ewald_6(k2, mode, phi0, phi1)
        integer, intent(in) :: mode
        real(8), intent(in) :: k2
        real(8), intent(out) :: phi0, phi1
        real(8) :: b1, b2, b3, erfc_b1, exp_b2
        real(8), parameter :: sqrt_pi = sqrt(const_pi)
        b2 = d2*k2*0.25d0
        b1 = sqrt(b2)
        b3 = b2*b1
        erfc_b1 = erfc(b1)
        exp_b2 = exp(-b2)
        select case (mode)
        case (0, 1, 2)
        phi0 = pf_phi0*(sqrt_pi*b3*erfc_b1+(0.5d0-b2)*exp_b2)
        case default ! -1 or -2
        phi0 = pf_phi0*(sqrt_pi*b3*erfc_b1+(0.5d0-b2)*exp_b2)
        phi1 = pf_phi1*(sqrt_pi*b1*erfc_b1-exp_b2)
        end select
    end
end


!returns reciprocal force field energy, called by Python
real(8) function reciprocal_energy(nname, name, na, pos, plv, nli, li, npar, par, nglo, glo) result(e)
    !DEC$ ATTRIBUTES DLLEXPORT :: RECIPROCAL_ENERGY
    use :: non_bonded_force_field
    implicit none
    integer, intent(in) :: na, nli, npar, nglo, nname, li(3, nli)
    real(8), intent(in) :: pos(3, na), plv(3, 3), par(npar, na, na), glo(nglo)
    character(nname), intent(in) :: name

    procedure(rho_lj_6_12), pointer :: func
    integer :: i, j, k
    real(8) :: lv(3), parglo(npar+nglo)

    func => rho_func_from_name(nname, name)
    e = 0d0
    parglo(npar+1:npar+nglo) = glo(:)

end

!accumulate reciprocal force field gradient, called by Python
subroutine reciprocal_gradient(nname, name, na, pos, plv, nli, li, npar, par, nglo, glo, delta2, grad, grad2)
    !DEC$ ATTRIBUTES DLLEXPORT :: RECIPROCAL_GRADIENT
    use :: non_bonded_force_field
    implicit none
    integer, intent(in) :: na, nli, npar, nglo, nname, li(3, nli)
    real(8), intent(in) :: pos(3, na), plv(3, 3), par(npar, na, na), glo(nglo), delta2
    character(nname), intent(in) :: name
    real(8), intent(inout) :: grad(3, na), grad2(3, 3)

    procedure(rho_lj_6_12), pointer :: func
    integer :: i, j, k, s
    real(8) :: lv(3), parglo(npar+nglo), g(3)

    func => rho_func_from_name(nname, name)
    parglo(npar+1:npar+nglo) = glo(:)

end

!accumulate reciprocal force field hessian, called by Python
subroutine reciprocal_hessian(nname, name, na, pos, plv, nli, li, npar, par, nglo, glo, delta2, hess)
    !DEC$ ATTRIBUTES DLLEXPORT :: RECIPROCAL_HESSIAN
    use :: non_bonded_force_field
    implicit none
    integer, intent(in) :: na, nli, npar, nglo, nname, li(3, nli)
    real(8), intent(in) :: pos(3, na), plv(3, 3), par(npar, na, na), glo(nglo), delta2
    character(nname), intent(in) :: name
    real(8), intent(inout) :: hess(3, na, 3, na)

    procedure(rho_lj_6_12), pointer :: func
    integer :: i, j, k
    real(8) :: lv(3), parglo(npar+nglo), h(3, 3)

    func => rho_func_from_name(nname, name)
    parglo(npar+1:npar+nglo) = glo(:)

end

!accumulate reciprocal force field dynamical, called by Python
subroutine reciprocal_dynamical(nname, name, na, pos, plv, nli, li, npar, par, nglo, glo, delta2, qv, dyna)
    !DEC$ ATTRIBUTES DLLEXPORT :: RECIPROCAL_DYNAMICAL
    use :: non_bonded_force_field
    implicit none
    integer, intent(in) :: na, nli, npar, nglo, nname, li(3, nli)
    real(8), intent(in) :: pos(3, na), plv(3, 3), par(npar, na, na), glo(nglo), delta2, qv(3)
    character(nname), intent(in) :: name
    complex(8), intent(inout) :: dyna(3, na, 3, na)

    procedure(rho_lj_6_12), pointer :: func
    integer :: i, j, k
    real(8) :: lv(3), parglo(npar+nglo), h(3, 3)
    complex(8), parameter :: i_const = (0d0, 1d0)
    complex(8) :: exp_dpi(2)

    func => rho_func_from_name(nname, name)
    parglo(npar+1:npar+nglo) = glo(:)

end

! get the indices of the lattice points within cutoff
! Inversion-symmetric points are omitted.
! If nli > nlimax, buffer size is too small.
subroutine lattice_points_within_cutoff(plv, cutoff, nlimax, nli, li) bind(c)
    !DEC$ ATTRIBUTES DLLEXPORT :: lattice_points_within_cutoff
    use :: common
    use :: iso_c_binding
    implicit none

    real(c_double), intent(in) :: plv(3, 3)        ! primitive lattice vectors
    real(c_double), intent(in) :: cutoff           ! cutoff length
    integer(c_int), intent(in) :: nlimax            ! buffer size
    integer(c_int), intent(out) :: nli              ! the number of lattice points within cutoff
    integer(c_int), intent(out) :: li(3, nlimax)    ! index of lattice points within cutoff

    integer :: i, j, k, cell_range(3)
    real(8) :: cutoff2, v(3)

    nli = 0
    cutoff2 = cutoff * cutoff
    cell_range = ewald_cell_range_within_cutoff(cutoff, plv)

    do k = 1, cell_range(3)
        do j = -cell_range(2), cell_range(2)
            do i = -cell_range(1), cell_range(1)
                v(:) = matmul(plv(:, :), (/ i, j, k /))
                if (sum(v(:)**2) <= cutoff2) then
                    nli = nli + 1
                    if (nli <= nlimax) then
                        li(:, nli) = (/ i, j, k /)
                    end if
                end if
            end do
        end do
    end do

    do j = 1, cell_range(2)
        do i = -cell_range(1), cell_range(1)
            v(:) = matmul(plv(:, :), (/ i, j, 0 /))
            if (sum(v(:)**2) <= cutoff2) then
                nli = nli + 1
                if (nli <= nlimax) then
                    li(:, nli) = (/ i, j, 0 /)
                end if
            end if
        end do
    end do

    do i = 1, cell_range(1)
        v(:) = matmul(plv(:, :), (/ i, 0, 0 /))
        if (sum(v(:)**2) <= cutoff2) then
            nli = nli + 1
            if (nli <= nlimax) then
                li(:, nli) = (/ i, 0, 0 /)
            end if
        end if
    end do

end
