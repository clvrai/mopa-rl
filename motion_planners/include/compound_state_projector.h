/// Provide a state projection for compound states.
/// This only uses the elements of the state that are expressed as a vector
/// of doubles.

#pragma once

#include <iostream>
#include <vector>

#include <ompl/base/ProjectionEvaluator.h>
#include <ompl/base/spaces/RealVectorStateProjections.h>
#include <Eigen/Dense>

class CompoundStateProjector
        : public ompl::base::RealVectorRandomLinearProjectionEvaluator {
  public:
    CompoundStateProjector(const ompl::base::CompoundStateSpace* space,
        std::shared_ptr<ompl::base::RealVectorStateSpace> real_space_,
        int dim_)
            : RealVectorRandomLinearProjectionEvaluator(real_space_, dim_),
            real_space(real_space_),
            dim(dim_) {
    }

    CompoundStateProjector(const ompl::base::CompoundStateSpace* space,
        std::shared_ptr<ompl::base::RealVectorStateSpace> real_space_,
        std::vector<double> cell_sizes)
            : RealVectorRandomLinearProjectionEvaluator(real_space_, cell_sizes),
            real_space(real_space_),
            dim(cell_sizes.size()) {}

    static std::shared_ptr<CompoundStateProjector> makeCompoundStateProjector(
        const ompl::base::CompoundStateSpace* space)
    {
        auto real_space = std::make_shared<ompl::base::RealVectorStateSpace>(
            space->getDimension());
        // real_space->setBounds(-3.14, 3.14);
        // real_space->setBounds(-1, 1);
        int dim = 3;
        std::vector<double> cell_sizes(dim, 0.1);
        auto csp(std::make_shared<CompoundStateProjector>(
            space, real_space, cell_sizes));
        return csp;
    }

    unsigned int getDimension() const override {
        return dim;
    }

    void project(
        const ompl::base::State* state,
        Eigen::Ref< Eigen::VectorXd > projection) const override
    {
        auto rv_state = getRealVectorState(state);

        // Use a real space projection
        ompl::base::RealVectorRandomLinearProjectionEvaluator::project(
            rv_state, projection);

        // Cleanup
        real_space->freeState(rv_state);
    }


  private:
    ompl::base::State* getRealVectorState(const ompl::base::State* state) const {
        // Create a real vector state
        std::vector<double> reals;
        real_space->copyToReals(reals, state);
        auto rv_state = real_space->allocState();
        for(size_t i=0; i < reals.size(); i++) {
            rv_state->as<ompl::base::RealVectorStateSpace::StateType>()
                ->values[i] = reals[i];
        }

        return rv_state;
    }

    std::shared_ptr<ompl::base::RealVectorStateSpace> real_space;
    const int dim;
};
