import CorrectionFactorMatrixMethod: SafeDoubleQuadRule, SafeDoubleNumQStrat

function BEAST.quadrule(
    op::BEAST.IntegralOperator,
    local_test_basis,
    local_trial_basis,
    test_id,
    test_element,
    trial_id,
    trial_element,
    quad_data,
    qs::SafeDoubleNumQStrat,
)
    return SafeDoubleQuadRule(quad_data[1][1, test_id], quad_data[2][1, trial_id])
end

function BEAST.quaddata(
    operator::BEAST.IntegralOperator,
    local_test_basis,
    local_trial_basis,
    test_elements,
    trial_elements,
    qs::SafeDoubleNumQStrat,
)
    test_quad_data  = BEAST.quadpoints(local_test_basis, test_elements, (qs.outer_rule,))
    trial_quad_data = BEAST.quadpoints(local_trial_basis, trial_elements, (qs.inner_rule,))

    return test_quad_data, trial_quad_data
end

function BEAST.momintegrals!(biop, tshs, bshs, tcell, bcell, z, strat::SafeDoubleQuadRule)
    igd = BEAST.Integrand(biop, tshs, bshs, tcell, bcell)
    womps = strat.outer_quad_points
    wimps = strat.inner_quad_points
    for womp in womps
        tgeo = womp.point
        tvals = womp.value
        M = length(tvals)
        jx = womp.weight
        for wimp in wimps
            bgeo = wimp.point
            bvals = wimp.value
            N = length(bvals)
            jy = wimp.weight

            j = jx * jy

            if !(bgeo.cart ≈ tgeo.cart)
                z1 = j * igd(tgeo, bgeo, tvals, bvals)
                for n in 1:N
                    for m in 1:M
                        z[m, n] += z1[m, n]
                    end
                end
            end
        end
    end

    return z
end
