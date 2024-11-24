#pragma once

#include <iostream>

struct disort_state;

//! \brief Report the disort atmosphere
/*!
 * \param ds Disort state
 * \param os Output stream
 */
void report_disort_atmosphere(disort_state const& ds, std::ostream& os);

//! \brief Report the disort output
/*!
 * \param ds Disort state
 * \param os Output stream
 */
void report_disort_output(disort_state const& ds, std::ostream& os);

//! \brief Report the disort boundary conditions
/*!
 * \param ds Disort state
 * \param os Output stream
 */
void report_disort_bcs(disort_state const& ds, std::ostream& os);

//! \brief Report the disort flags
/*!
 * \param ds Disort state
 * \param os Output stream
 */
void report_disort_flags(disort_state const& ds, std::ostream& os);

//! \brief Report the disort state
/*!
 * \param ds Disort state
 * \return String representation of the disort state
 */
std::string report_disort(disort_state const& ds);
