function _cuda_combined_kernel!(
    a_re, a_im, c_re, c_im,
    face_vertices,
    normals,
    areas,
    faces,
    curls,
    test_indices,
    trial_indices,
    rule_points,
    rule_weights,
    k,
    p1_dof_count,
    dp0_dof_count,
    face_count,
    rule_count,
    total_pairs,
    skip_adjacent,
    coupling_scale,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
)
    pair = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    four_pi = typeof(k)(12.566370614359172)

    while pair <= total_pairs
        test_loop_index = ((pair - 1) % length(test_indices)) + 1
        trial_loop_index = ((pair - 1) ÷ length(test_indices)) + 1
        test_index = test_indices[test_loop_index]
        trial_index = trial_indices[trial_loop_index]

        t1 = faces[test_index]
        t2 = faces[test_index + face_count]
        t3 = faces[test_index + 2 * face_count]
        r1 = faces[trial_index]
        r2 = faces[trial_index + face_count]
        r3 = faces[trial_index + 2 * face_count]

        adjacent = t1 == r1 || t1 == r2 || t1 == r3 ||
            t2 == r1 || t2 == r2 || t2 == r3 ||
            t3 == r1 || t3 == r2 || t3 == r3

        if !skip_adjacent || !adjacent
            tv1x = face_vertices[test_index]
            tv1y = face_vertices[test_index + face_count]
            tv1z = face_vertices[test_index + 2 * face_count]
            tv2x = face_vertices[test_index + 3 * face_count]
            tv2y = face_vertices[test_index + 4 * face_count]
            tv2z = face_vertices[test_index + 5 * face_count]
            tv3x = face_vertices[test_index + 6 * face_count]
            tv3y = face_vertices[test_index + 7 * face_count]
            tv3z = face_vertices[test_index + 8 * face_count]

            rv1x = trial_sign_x * face_vertices[trial_index]
            rv1y = trial_sign_y * face_vertices[trial_index + face_count]
            rv1z = trial_sign_z * face_vertices[trial_index + 2 * face_count]
            rv2x = trial_sign_x * face_vertices[trial_index + 3 * face_count]
            rv2y = trial_sign_y * face_vertices[trial_index + 4 * face_count]
            rv2z = trial_sign_z * face_vertices[trial_index + 5 * face_count]
            rv3x = trial_sign_x * face_vertices[trial_index + 6 * face_count]
            rv3y = trial_sign_y * face_vertices[trial_index + 7 * face_count]
            rv3z = trial_sign_z * face_vertices[trial_index + 8 * face_count]

            tnx = normals[test_index]
            tny = normals[test_index + face_count]
            tnz = normals[test_index + 2 * face_count]
            rnx = trial_sign_x * normals[trial_index]
            rny = trial_sign_y * normals[trial_index + face_count]
            rnz = trial_sign_z * normals[trial_index + 2 * face_count]
            normal_product = tnx * rnx + tny * rny + tnz * rnz

            tc11 = curls[test_index]
            tc12 = curls[test_index + face_count]
            tc13 = curls[test_index + 2 * face_count]
            tc21 = curls[test_index + 3 * face_count]
            tc22 = curls[test_index + 4 * face_count]
            tc23 = curls[test_index + 5 * face_count]
            tc31 = curls[test_index + 6 * face_count]
            tc32 = curls[test_index + 7 * face_count]
            tc33 = curls[test_index + 8 * face_count]

            rc11 = trial_curl_sign_x * curls[trial_index]
            rc12 = trial_curl_sign_y * curls[trial_index + face_count]
            rc13 = trial_curl_sign_z * curls[trial_index + 2 * face_count]
            rc21 = trial_curl_sign_x * curls[trial_index + 3 * face_count]
            rc22 = trial_curl_sign_y * curls[trial_index + 4 * face_count]
            rc23 = trial_curl_sign_z * curls[trial_index + 5 * face_count]
            rc31 = trial_curl_sign_x * curls[trial_index + 6 * face_count]
            rc32 = trial_curl_sign_y * curls[trial_index + 7 * face_count]
            rc33 = trial_curl_sign_z * curls[trial_index + 8 * face_count]

            jac_scale = typeof(k)(4) * areas[test_index] * areas[trial_index]

            c1_re = zero(k)
            c1_im = zero(k)
            c2_re = zero(k)
            c2_im = zero(k)
            c3_re = zero(k)
            c3_im = zero(k)
            a11_re = zero(k)
            a11_im = zero(k)
            a12_re = zero(k)
            a12_im = zero(k)
            a13_re = zero(k)
            a13_im = zero(k)
            a21_re = zero(k)
            a21_im = zero(k)
            a22_re = zero(k)
            a22_im = zero(k)
            a23_re = zero(k)
            a23_im = zero(k)
            a31_re = zero(k)
            a31_im = zero(k)
            a32_re = zero(k)
            a32_im = zero(k)
            a33_re = zero(k)
            a33_im = zero(k)
            inverse_k = coupling_scale
            for tq in 1:rule_count
                txi = rule_points[tq]
                teta = rule_points[tq + rule_count]
                tw = rule_weights[tq]
                tv1 = one(k) - txi - teta
                tv2 = txi
                tv3 = teta
                x = tv1 * tv1x + tv2 * tv2x + tv3 * tv3x
                y = tv1 * tv1y + tv2 * tv2y + tv3 * tv3y
                z = tv1 * tv1z + tv2 * tv2z + tv3 * tv3z

                for rq in 1:rule_count
                    rxi = rule_points[rq]
                    reta = rule_points[rq + rule_count]
                    rw = rule_weights[rq]
                    rv1 = one(k) - rxi - reta
                    rv2 = rxi
                    rv3 = reta
                    sx = rv1 * rv1x + rv2 * rv2x + rv3 * rv3x
                    sy = rv1 * rv1y + rv2 * rv2y + rv3 * rv3y
                    sz = rv1 * rv1z + rv2 * rv2z + rv3 * rv3z

                    dx = sx - x
                    dy = sy - y
                    dz = sz - z
                    radius = sqrt(dx * dx + dy * dy + dz * dz)

                    if radius > zero(k)
                        inv_radius = one(k) / radius
                        phase = k * radius
                        green_scale = inv_radius / four_pi
                        green_re = cos(phase) * green_scale
                        green_im = sin(phase) * green_scale
                        weight = tw * rw * jac_scale
                        weighted_re = green_re * weight
                        weighted_im = green_im * weight

                        source_projection = (dx * rnx + dy * rny + dz * rnz) * inv_radius
                        test_projection = -(dx * tnx + dy * tny + dz * tnz) * inv_radius
                        factor_re = -inv_radius
                        factor_im = k
                        deriv_re = green_re * factor_re - green_im * factor_im
                        deriv_im = green_re * factor_im + green_im * factor_re
                        dlp_value_re = deriv_re * source_projection * weight
                        dlp_value_im = deriv_im * source_projection * weight
                        adj_value_re = deriv_re * test_projection * weight
                        adj_value_im = deriv_im * test_projection * weight

                        c1_re += tv1 * (weighted_re - inverse_k * adj_value_im)
                        c1_im += tv1 * (weighted_im + inverse_k * adj_value_re)
                        c2_re += tv2 * (weighted_re - inverse_k * adj_value_im)
                        c2_im += tv2 * (weighted_im + inverse_k * adj_value_re)
                        c3_re += tv3 * (weighted_re - inverse_k * adj_value_im)
                        c3_im += tv3 * (weighted_im + inverse_k * adj_value_re)
                        h11 = (tc11 * rc11 + tc12 * rc12 + tc13 * rc13) - k * k * tv1 * rv1 * normal_product
                        h12 = (tc11 * rc21 + tc12 * rc22 + tc13 * rc23) - k * k * tv1 * rv2 * normal_product
                        h13 = (tc11 * rc31 + tc12 * rc32 + tc13 * rc33) - k * k * tv1 * rv3 * normal_product
                        h21 = (tc21 * rc11 + tc22 * rc12 + tc23 * rc13) - k * k * tv2 * rv1 * normal_product
                        h22 = (tc21 * rc21 + tc22 * rc22 + tc23 * rc23) - k * k * tv2 * rv2 * normal_product
                        h23 = (tc21 * rc31 + tc22 * rc32 + tc23 * rc33) - k * k * tv2 * rv3 * normal_product
                        h31 = (tc31 * rc11 + tc32 * rc12 + tc33 * rc13) - k * k * tv3 * rv1 * normal_product
                        h32 = (tc31 * rc21 + tc32 * rc22 + tc33 * rc23) - k * k * tv3 * rv2 * normal_product
                        h33 = (tc31 * rc31 + tc32 * rc32 + tc33 * rc33) - k * k * tv3 * rv3 * normal_product

                        a11_re += -tv1 * rv1 * dlp_value_re - inverse_k * h11 * weighted_im
                        a11_im += -tv1 * rv1 * dlp_value_im + inverse_k * h11 * weighted_re
                        a12_re += -tv1 * rv2 * dlp_value_re - inverse_k * h12 * weighted_im
                        a12_im += -tv1 * rv2 * dlp_value_im + inverse_k * h12 * weighted_re
                        a13_re += -tv1 * rv3 * dlp_value_re - inverse_k * h13 * weighted_im
                        a13_im += -tv1 * rv3 * dlp_value_im + inverse_k * h13 * weighted_re
                        a21_re += -tv2 * rv1 * dlp_value_re - inverse_k * h21 * weighted_im
                        a21_im += -tv2 * rv1 * dlp_value_im + inverse_k * h21 * weighted_re
                        a22_re += -tv2 * rv2 * dlp_value_re - inverse_k * h22 * weighted_im
                        a22_im += -tv2 * rv2 * dlp_value_im + inverse_k * h22 * weighted_re
                        a23_re += -tv2 * rv3 * dlp_value_re - inverse_k * h23 * weighted_im
                        a23_im += -tv2 * rv3 * dlp_value_im + inverse_k * h23 * weighted_re
                        a31_re += -tv3 * rv1 * dlp_value_re - inverse_k * h31 * weighted_im
                        a31_im += -tv3 * rv1 * dlp_value_im + inverse_k * h31 * weighted_re
                        a32_re += -tv3 * rv2 * dlp_value_re - inverse_k * h32 * weighted_im
                        a32_im += -tv3 * rv2 * dlp_value_im + inverse_k * h32 * weighted_re
                        a33_re += -tv3 * rv3 * dlp_value_re - inverse_k * h33 * weighted_im
                        a33_im += -tv3 * rv3 * dlp_value_im + inverse_k * h33 * weighted_re
                    end
                end
            end

            _cuda_atomic_add!(c_re, t1 + (trial_index - 1) * p1_dof_count, c1_re)
            _cuda_atomic_add!(c_im, t1 + (trial_index - 1) * p1_dof_count, c1_im)
            _cuda_atomic_add!(c_re, t2 + (trial_index - 1) * p1_dof_count, c2_re)
            _cuda_atomic_add!(c_im, t2 + (trial_index - 1) * p1_dof_count, c2_im)
            _cuda_atomic_add!(c_re, t3 + (trial_index - 1) * p1_dof_count, c3_re)
            _cuda_atomic_add!(c_im, t3 + (trial_index - 1) * p1_dof_count, c3_im)
            _cuda_atomic_add!(a_re, t1 + (r1 - 1) * p1_dof_count, a11_re)
            _cuda_atomic_add!(a_im, t1 + (r1 - 1) * p1_dof_count, a11_im)
            _cuda_atomic_add!(a_re, t1 + (r2 - 1) * p1_dof_count, a12_re)
            _cuda_atomic_add!(a_im, t1 + (r2 - 1) * p1_dof_count, a12_im)
            _cuda_atomic_add!(a_re, t1 + (r3 - 1) * p1_dof_count, a13_re)
            _cuda_atomic_add!(a_im, t1 + (r3 - 1) * p1_dof_count, a13_im)
            _cuda_atomic_add!(a_re, t2 + (r1 - 1) * p1_dof_count, a21_re)
            _cuda_atomic_add!(a_im, t2 + (r1 - 1) * p1_dof_count, a21_im)
            _cuda_atomic_add!(a_re, t2 + (r2 - 1) * p1_dof_count, a22_re)
            _cuda_atomic_add!(a_im, t2 + (r2 - 1) * p1_dof_count, a22_im)
            _cuda_atomic_add!(a_re, t2 + (r3 - 1) * p1_dof_count, a23_re)
            _cuda_atomic_add!(a_im, t2 + (r3 - 1) * p1_dof_count, a23_im)
            _cuda_atomic_add!(a_re, t3 + (r1 - 1) * p1_dof_count, a31_re)
            _cuda_atomic_add!(a_im, t3 + (r1 - 1) * p1_dof_count, a31_im)
            _cuda_atomic_add!(a_re, t3 + (r2 - 1) * p1_dof_count, a32_re)
            _cuda_atomic_add!(a_im, t3 + (r2 - 1) * p1_dof_count, a32_im)
            _cuda_atomic_add!(a_re, t3 + (r3 - 1) * p1_dof_count, a33_re)
            _cuda_atomic_add!(a_im, t3 + (r3 - 1) * p1_dof_count, a33_im)

        end

        pair += stride
    end

    return nothing
end

function _cuda_combined_fused_kernel!(
    a_re, a_im, c_re, c_im,
    face_vertices,
    normals,
    areas,
    faces,
    curls,
    test_indices,
    trial_indices,
    rule_points,
    rule_weights,
    k,
    p1_dof_count,
    dp0_dof_count,
    face_count,
    rule_count,
    total_pairs,
    skip_adjacent,
    coupling_scale,
    transforms,
)
    pair = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    four_pi = typeof(k)(12.566370614359172)

    while pair <= total_pairs
        test_loop_index = ((pair - 1) % length(test_indices)) + 1
        trial_loop_index = ((pair - 1) ÷ length(test_indices)) + 1
        test_index = test_indices[test_loop_index]
        trial_index = trial_indices[trial_loop_index]

        t1 = faces[test_index]
        t2 = faces[test_index + face_count]
        t3 = faces[test_index + 2 * face_count]
        r1 = faces[trial_index]
        r2 = faces[trial_index + face_count]
        r3 = faces[trial_index + 2 * face_count]

        adjacent = t1 == r1 || t1 == r2 || t1 == r3 ||
            t2 == r1 || t2 == r2 || t2 == r3 ||
            t3 == r1 || t3 == r2 || t3 == r3

        if !skip_adjacent || !adjacent
            c1_re = zero(k)
            c1_im = zero(k)
            c2_re = zero(k)
            c2_im = zero(k)
            c3_re = zero(k)
            c3_im = zero(k)
            a11_re = zero(k)
            a11_im = zero(k)
            a12_re = zero(k)
            a12_im = zero(k)
            a13_re = zero(k)
            a13_im = zero(k)
            a21_re = zero(k)
            a21_im = zero(k)
            a22_re = zero(k)
            a22_im = zero(k)
            a23_re = zero(k)
            a23_im = zero(k)
            a31_re = zero(k)
            a31_im = zero(k)
            a32_re = zero(k)
            a32_im = zero(k)
            a33_re = zero(k)
            a33_im = zero(k)
            inverse_k = coupling_scale
            for transform in transforms
            trial_sign_x = transform[1]
            trial_sign_y = transform[2]
            trial_sign_z = transform[3]
            trial_curl_sign_x = transform[4]
            trial_curl_sign_y = transform[5]
            trial_curl_sign_z = transform[6]
            tv1x = face_vertices[test_index]
            tv1y = face_vertices[test_index + face_count]
            tv1z = face_vertices[test_index + 2 * face_count]
            tv2x = face_vertices[test_index + 3 * face_count]
            tv2y = face_vertices[test_index + 4 * face_count]
            tv2z = face_vertices[test_index + 5 * face_count]
            tv3x = face_vertices[test_index + 6 * face_count]
            tv3y = face_vertices[test_index + 7 * face_count]
            tv3z = face_vertices[test_index + 8 * face_count]

            rv1x = trial_sign_x * face_vertices[trial_index]
            rv1y = trial_sign_y * face_vertices[trial_index + face_count]
            rv1z = trial_sign_z * face_vertices[trial_index + 2 * face_count]
            rv2x = trial_sign_x * face_vertices[trial_index + 3 * face_count]
            rv2y = trial_sign_y * face_vertices[trial_index + 4 * face_count]
            rv2z = trial_sign_z * face_vertices[trial_index + 5 * face_count]
            rv3x = trial_sign_x * face_vertices[trial_index + 6 * face_count]
            rv3y = trial_sign_y * face_vertices[trial_index + 7 * face_count]
            rv3z = trial_sign_z * face_vertices[trial_index + 8 * face_count]

            tnx = normals[test_index]
            tny = normals[test_index + face_count]
            tnz = normals[test_index + 2 * face_count]
            rnx = trial_sign_x * normals[trial_index]
            rny = trial_sign_y * normals[trial_index + face_count]
            rnz = trial_sign_z * normals[trial_index + 2 * face_count]
            normal_product = tnx * rnx + tny * rny + tnz * rnz

            tc11 = curls[test_index]
            tc12 = curls[test_index + face_count]
            tc13 = curls[test_index + 2 * face_count]
            tc21 = curls[test_index + 3 * face_count]
            tc22 = curls[test_index + 4 * face_count]
            tc23 = curls[test_index + 5 * face_count]
            tc31 = curls[test_index + 6 * face_count]
            tc32 = curls[test_index + 7 * face_count]
            tc33 = curls[test_index + 8 * face_count]

            rc11 = trial_curl_sign_x * curls[trial_index]
            rc12 = trial_curl_sign_y * curls[trial_index + face_count]
            rc13 = trial_curl_sign_z * curls[trial_index + 2 * face_count]
            rc21 = trial_curl_sign_x * curls[trial_index + 3 * face_count]
            rc22 = trial_curl_sign_y * curls[trial_index + 4 * face_count]
            rc23 = trial_curl_sign_z * curls[trial_index + 5 * face_count]
            rc31 = trial_curl_sign_x * curls[trial_index + 6 * face_count]
            rc32 = trial_curl_sign_y * curls[trial_index + 7 * face_count]
            rc33 = trial_curl_sign_z * curls[trial_index + 8 * face_count]

            jac_scale = typeof(k)(4) * areas[test_index] * areas[trial_index]

            for tq in 1:rule_count
                txi = rule_points[tq]
                teta = rule_points[tq + rule_count]
                tw = rule_weights[tq]
                tv1 = one(k) - txi - teta
                tv2 = txi
                tv3 = teta
                x = tv1 * tv1x + tv2 * tv2x + tv3 * tv3x
                y = tv1 * tv1y + tv2 * tv2y + tv3 * tv3y
                z = tv1 * tv1z + tv2 * tv2z + tv3 * tv3z

                for rq in 1:rule_count
                    rxi = rule_points[rq]
                    reta = rule_points[rq + rule_count]
                    rw = rule_weights[rq]
                    rv1 = one(k) - rxi - reta
                    rv2 = rxi
                    rv3 = reta
                    sx = rv1 * rv1x + rv2 * rv2x + rv3 * rv3x
                    sy = rv1 * rv1y + rv2 * rv2y + rv3 * rv3y
                    sz = rv1 * rv1z + rv2 * rv2z + rv3 * rv3z

                    dx = sx - x
                    dy = sy - y
                    dz = sz - z
                    radius = sqrt(dx * dx + dy * dy + dz * dz)

                    if radius > zero(k)
                        inv_radius = one(k) / radius
                        phase = k * radius
                        green_scale = inv_radius / four_pi
                        green_re = cos(phase) * green_scale
                        green_im = sin(phase) * green_scale
                        weight = tw * rw * jac_scale
                        weighted_re = green_re * weight
                        weighted_im = green_im * weight

                        source_projection = (dx * rnx + dy * rny + dz * rnz) * inv_radius
                        test_projection = -(dx * tnx + dy * tny + dz * tnz) * inv_radius
                        factor_re = -inv_radius
                        factor_im = k
                        deriv_re = green_re * factor_re - green_im * factor_im
                        deriv_im = green_re * factor_im + green_im * factor_re
                        dlp_value_re = deriv_re * source_projection * weight
                        dlp_value_im = deriv_im * source_projection * weight
                        adj_value_re = deriv_re * test_projection * weight
                        adj_value_im = deriv_im * test_projection * weight

                        c1_re += tv1 * (weighted_re - inverse_k * adj_value_im)
                        c1_im += tv1 * (weighted_im + inverse_k * adj_value_re)
                        c2_re += tv2 * (weighted_re - inverse_k * adj_value_im)
                        c2_im += tv2 * (weighted_im + inverse_k * adj_value_re)
                        c3_re += tv3 * (weighted_re - inverse_k * adj_value_im)
                        c3_im += tv3 * (weighted_im + inverse_k * adj_value_re)
                        h11 = (tc11 * rc11 + tc12 * rc12 + tc13 * rc13) - k * k * tv1 * rv1 * normal_product
                        h12 = (tc11 * rc21 + tc12 * rc22 + tc13 * rc23) - k * k * tv1 * rv2 * normal_product
                        h13 = (tc11 * rc31 + tc12 * rc32 + tc13 * rc33) - k * k * tv1 * rv3 * normal_product
                        h21 = (tc21 * rc11 + tc22 * rc12 + tc23 * rc13) - k * k * tv2 * rv1 * normal_product
                        h22 = (tc21 * rc21 + tc22 * rc22 + tc23 * rc23) - k * k * tv2 * rv2 * normal_product
                        h23 = (tc21 * rc31 + tc22 * rc32 + tc23 * rc33) - k * k * tv2 * rv3 * normal_product
                        h31 = (tc31 * rc11 + tc32 * rc12 + tc33 * rc13) - k * k * tv3 * rv1 * normal_product
                        h32 = (tc31 * rc21 + tc32 * rc22 + tc33 * rc23) - k * k * tv3 * rv2 * normal_product
                        h33 = (tc31 * rc31 + tc32 * rc32 + tc33 * rc33) - k * k * tv3 * rv3 * normal_product

                        a11_re += -tv1 * rv1 * dlp_value_re - inverse_k * h11 * weighted_im
                        a11_im += -tv1 * rv1 * dlp_value_im + inverse_k * h11 * weighted_re
                        a12_re += -tv1 * rv2 * dlp_value_re - inverse_k * h12 * weighted_im
                        a12_im += -tv1 * rv2 * dlp_value_im + inverse_k * h12 * weighted_re
                        a13_re += -tv1 * rv3 * dlp_value_re - inverse_k * h13 * weighted_im
                        a13_im += -tv1 * rv3 * dlp_value_im + inverse_k * h13 * weighted_re
                        a21_re += -tv2 * rv1 * dlp_value_re - inverse_k * h21 * weighted_im
                        a21_im += -tv2 * rv1 * dlp_value_im + inverse_k * h21 * weighted_re
                        a22_re += -tv2 * rv2 * dlp_value_re - inverse_k * h22 * weighted_im
                        a22_im += -tv2 * rv2 * dlp_value_im + inverse_k * h22 * weighted_re
                        a23_re += -tv2 * rv3 * dlp_value_re - inverse_k * h23 * weighted_im
                        a23_im += -tv2 * rv3 * dlp_value_im + inverse_k * h23 * weighted_re
                        a31_re += -tv3 * rv1 * dlp_value_re - inverse_k * h31 * weighted_im
                        a31_im += -tv3 * rv1 * dlp_value_im + inverse_k * h31 * weighted_re
                        a32_re += -tv3 * rv2 * dlp_value_re - inverse_k * h32 * weighted_im
                        a32_im += -tv3 * rv2 * dlp_value_im + inverse_k * h32 * weighted_re
                        a33_re += -tv3 * rv3 * dlp_value_re - inverse_k * h33 * weighted_im
                        a33_im += -tv3 * rv3 * dlp_value_im + inverse_k * h33 * weighted_re
                    end
                end
            end

            end # sequential image transforms, one scatter
            _cuda_atomic_add!(c_re, t1 + (trial_index - 1) * p1_dof_count, c1_re)
            _cuda_atomic_add!(c_im, t1 + (trial_index - 1) * p1_dof_count, c1_im)
            _cuda_atomic_add!(c_re, t2 + (trial_index - 1) * p1_dof_count, c2_re)
            _cuda_atomic_add!(c_im, t2 + (trial_index - 1) * p1_dof_count, c2_im)
            _cuda_atomic_add!(c_re, t3 + (trial_index - 1) * p1_dof_count, c3_re)
            _cuda_atomic_add!(c_im, t3 + (trial_index - 1) * p1_dof_count, c3_im)
            _cuda_atomic_add!(a_re, t1 + (r1 - 1) * p1_dof_count, a11_re)
            _cuda_atomic_add!(a_im, t1 + (r1 - 1) * p1_dof_count, a11_im)
            _cuda_atomic_add!(a_re, t1 + (r2 - 1) * p1_dof_count, a12_re)
            _cuda_atomic_add!(a_im, t1 + (r2 - 1) * p1_dof_count, a12_im)
            _cuda_atomic_add!(a_re, t1 + (r3 - 1) * p1_dof_count, a13_re)
            _cuda_atomic_add!(a_im, t1 + (r3 - 1) * p1_dof_count, a13_im)
            _cuda_atomic_add!(a_re, t2 + (r1 - 1) * p1_dof_count, a21_re)
            _cuda_atomic_add!(a_im, t2 + (r1 - 1) * p1_dof_count, a21_im)
            _cuda_atomic_add!(a_re, t2 + (r2 - 1) * p1_dof_count, a22_re)
            _cuda_atomic_add!(a_im, t2 + (r2 - 1) * p1_dof_count, a22_im)
            _cuda_atomic_add!(a_re, t2 + (r3 - 1) * p1_dof_count, a23_re)
            _cuda_atomic_add!(a_im, t2 + (r3 - 1) * p1_dof_count, a23_im)
            _cuda_atomic_add!(a_re, t3 + (r1 - 1) * p1_dof_count, a31_re)
            _cuda_atomic_add!(a_im, t3 + (r1 - 1) * p1_dof_count, a31_im)
            _cuda_atomic_add!(a_re, t3 + (r2 - 1) * p1_dof_count, a32_re)
            _cuda_atomic_add!(a_im, t3 + (r2 - 1) * p1_dof_count, a32_im)
            _cuda_atomic_add!(a_re, t3 + (r3 - 1) * p1_dof_count, a33_re)
            _cuda_atomic_add!(a_im, t3 + (r3 - 1) * p1_dof_count, a33_im)

        end

        pair += stride
    end

    return nothing
end

# Coupled Burton-Miller A=-D+alpha*H, C=S+alpha*adjD before weighted identities.
function _launch_combined!(ar, ai, cr, ci, cache, k, transforms; skip_adjacent, fused=false, max_registers=0,
                          coupling_scale=inv(k))
    count = length(cache.element_indices)^2
    args = (ar, ai, cr, ci, cache.face_vertices, cache.normals, cache.areas,
        cache.faces, cache.curls, cache.test_indices, cache.trial_indices,
        cache.rule_points, cache.rule_weights, k, size(ar, 1), size(cr, 2),
        cache.face_count, cache.rule_count, count, skip_adjacent, coupling_scale)
    signs = Tuple((T = typeof(k); (T.(t.signs)..., T.(t.determinant .* t.signs)...)) for t in transforms)
    if fused && max_registers > 0
        CUDA.@cuda maxregs=max_registers threads=128 blocks=min(cld(count,128),65535) _cuda_combined_fused_kernel!(args..., signs)
    elseif fused
        CUDA.@cuda threads=128 blocks=min(cld(count,128),65535) _cuda_combined_fused_kernel!(args..., signs)
    else
        for sign in signs
            CUDA.@cuda threads=128 blocks=min(cld(count,128),65535) _cuda_combined_kernel!(args..., sign...)
        end
    end
    CUDA.synchronize()
end

function _cuda_combined_correction_kernel!(ar, ai, cr, ci, rows, cols, dpcols, s, adj, d, h, ik, n, count)
    pair = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while pair <= count
        for i in 1:3
            row = rows[pair + (i-1)*count]
            pos = pair + (i-1)*count
            index = row + (dpcols[pair]-1)*n
            _cuda_atomic_add!(cr, index, real(s[pos]) - ik*imag(adj[pos]))
            _cuda_atomic_add!(ci, index, imag(s[pos]) + ik*real(adj[pos]))
            for j in 1:3
                col = cols[pair + (j-1)*count]
                pos = pair + ((j-1)*3+i-1)*count
                _cuda_bm_add_lhs!(ar, ai, row+(col-1)*n, real(d[pos]), imag(d[pos]), real(h[pos]), imag(h[pos]), ik)
            end
        end
        pair += stride
    end
    return nothing
end

function _scatter_cuda_bm_blocks!(ar, ai, cr, ci, ::Val{:combined}, blocks, cache, k; rhs_only=false, coupling_scale=inv(k))
    count = cache.pair_count
    count == 0 && return 0
    CUDA.@cuda threads=128 blocks=min(cld(count,128),65535) _cuda_combined_correction_kernel!(
        ar, ai, cr, ci, cache.p1_rows, cache.p1_cols, cache.dp0_cols,
        blocks.slp, blocks.adjoint, blocks.dlp, blocks.hypersingular, coupling_scale, size(ar,1), count)
    CUDA.synchronize()
    return count
end

function assemble_coupled_burton_miller_cuda(mesh::BoundaryMesh{T}, prepared, k; fused=true, max_registers=0,
                                             coupling_cap::Real=zero(T)) where T
    k = outgoing_wavenumber(k)
    # Signed like k, as inv(k) was; cap = 0 reproduces inv(k) exactly.
    coupling_scale = copysign(burton_miller_coupling_scale(k, coupling_cap), k)
    n = prepared.p1.global_dof_count
    m = prepared.dp0.global_dof_count
    astorage = CUDA.zeros(T, 2, n, n)
    cstorage = nothing
    ok = false
    try
        cstorage = CUDA.zeros(T, 2, n, m)
        ar, ai = view(astorage,1,:,:), view(astorage,2,:,:)
        cr, ci = view(cstorage,1,:,:), view(cstorage,2,:,:)
        _launch_combined!(ar,ai,cr,ci,prepared.device_cache,k,
            symmetry_transforms(:off; include_identity=true); skip_adjacent=true, coupling_scale=coupling_scale)
        images = symmetry_image_transforms(prepared.symmetry_mode)
        isempty(images) || _launch_combined!(ar,ai,cr,ci,prepared.device_cache,k,images; skip_adjacent=false,fused=fused,max_registers=max_registers,
            coupling_scale=coupling_scale)
        add_cuda_bm_singular_corrections!(ar,ai,cr,ci,Val(:combined),mesh,k,
            prepared.singular_cache,prepared.device_singular_cache,prepared.device_cache;
            coupling_scale=coupling_scale)
        add_cuda_bm_image_corrections!(ar,ai,cr,ci,Val(:combined),mesh,k,prepared.rule,
            prepared.device_image_singular_cache,prepared.device_cache;
            coupling_scale=coupling_scale)
        a = reshape(reinterpret(Complex{T}, astorage),n,n)
        c = reshape(reinterpret(Complex{T}, cstorage),n,m)
        weights = CUDA.CuArray(p1_symmetry_orbit_weights(mesh, prepared.symmetry_mode))
        try
            a .*= weights
            c .*= weights
            a .+= T(0.5) .* prepared.device_identity_cache.identity_p1_p1
            c .+= Complex{T}(0,T(0.5)*coupling_scale) .* prepared.device_identity_cache.identity_p1_dp0
            CUDA.synchronize()
        finally
            CUDA.unsafe_free!(weights)
        end
        ok = true
        return (a=a,c=c)
    finally
        if !ok
            CUDA.unsafe_free!(astorage)
            cstorage === nothing || CUDA.unsafe_free!(cstorage)
        end
    end
end

# One owner per output entry: CSC interface adjacency, no atomics.
function _cuda_combined_projection_kernel!(out, c, colptr, rowval, values, n, total)
    index = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x
    while index <= total
        row = (index-1)%n+1
        col = (index-1) ÷ n+1
        value = zero(eltype(out))
        for pos in colptr[col]:(colptr[col+1]-1)
            value += c[row+(rowval[pos]-1)*n]*values[pos]
        end
        out[index] = value
        index += stride
    end
    return nothing
end

function build_cuda_combined_bem_blocks(operators, q, motion, prescribed)
    a,c = operators.a, operators.c
    interface = motion_block = prescribed_rhs = nothing
    success = false
    try
        if q isa NamedTuple
            interface = CUDA.zeros(eltype(c),size(c,1),q.ncols)
            isempty(interface) || CUDA.@cuda threads=256 blocks=min(cld(length(interface),256),65535) _cuda_combined_projection_kernel!(
                interface,c,q.colptr,q.rowval,q.nzval,size(c,1),length(interface))
        else
            error("Combined CUDA BEM blocks require the cached sparse interface map.")
        end
        if size(motion,2)>0
            d = CUDA.CuArray(Matrix(motion))
            try; motion_block = c*d; finally; CUDA.unsafe_free!(d); end
        end
        if size(prescribed,2)>0
            d = CUDA.CuArray(Matrix(prescribed))
            try; prescribed_rhs = -(c*d); finally; CUDA.unsafe_free!(d); end
        end
        CUDA.synchronize()
        success = true
        return (bem_lhs=a,bem_rhs_operator=nothing,bem_interface_block=interface,
            bem_motion_block=motion_block,bem_prescribed_rhs=prescribed_rhs)
    finally
        CUDA.unsafe_free!(c)
        if !success
            CUDA.unsafe_free!(a)
            for array in (interface,motion_block,prescribed_rhs)
                array === nothing || CUDA.unsafe_free!(array)
            end
        end
    end
end

"""Job-owned CSC projection; matrix values include the interface orientation weights."""
function build_cuda_bem_flux_cache(q)
    colptr = rowval = nzval = nothing
    try
        colptr = CUDA.CuArray(Int32.(q.colptr))
        rowval = CUDA.CuArray(Int32.(q.rowval))
        nzval = CUDA.CuArray(complex.(q.nzval))
        return (; colptr, rowval, nzval, ncols=size(q, 2))
    catch
        for array in (colptr, rowval, nzval)
            array === nothing || CUDA.unsafe_free!(array)
        end
        rethrow()
    end
end

function release_cuda_bem_flux_cache!(q)
    for array in (q.colptr, q.rowval, q.nzval)
        CUDA.unsafe_free!(array)
    end
    return nothing
end
