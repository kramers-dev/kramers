# Errata found while building the handbook

Misprints in published equations, found when an entry was implemented and checked. Each was verified numerically here; none of them affects the papers' main results unless stated. We have not found published errata for them. If you are an author and we are wrong, please say so in an issue; we will correct this page and credit you.

## K0002. Exact mean first-passage time in one dimension (quadrature formula)

Erratum found while implementing this entry. In Eq. (7.7) of Haenggi, Talkner and Borkovec as printed (journal scan, p. 291), the inner integral in the numerator of the first term has exp[U(z)/kBT]; it must be exp[-U(z)/kBT]. Computed check: for U = x^3/3 - x on [-2, 1] with kT = D = 1, the printed form gives t(b) = 2.511 instead of 0, while the corrected form agrees with this entry to 1e-10. A flat potential cannot reveal the difference. No erratum was found in Crossref.

## K0011. Variance of the passage time through a noisy saddle-node, any bias and thresholds

Two misprints found while implementing this entry, both checked numerically. (1) Eq. (3.2) as printed has the prefactor (9/D)^(2/3); the correct prefactor is 2 (9/D)^(2/3). As printed it gives exactly half of the paper's own Eq. (3.5) at beta = 0 (computed: ratio 2.0000 at beta = 0, -0.5 and 0.8). (2) In Eq. (A.3) the middle integrand must be exp(+V(z)/D): with the printed minus sign a linear potential V = -mu x gives 8.3e3 instead of the exact 2 D L / mu^3 = 0.728 (mu = 1.3, D = 0.4, L = 2), while the plus sign reproduces it. Eqs. (A.1), (A.2) and (A.4) are consistent with each other and with this entry.
