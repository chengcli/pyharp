
namespace harp {

class AbsorberRFMImpl : public AttenuatorImpl,
                        public torch::nn::Cloneable<AbsorberRFMImpl> {
 public:
  //! extinction x-section + single scattering albedo + phase function moments
  //! (batch, specs, temps, levels, comps)
  torch::Tensor kdata;

  //! scale the atmospheric variables to the standard grid
  AtmToStandardGrid scale_grid;

  //! Constructor to initialize the layer
  AbsorberRFMImpl() = default;
  explicit AbsorberRFMImpl(AttenuatorOptions const& options_);
  void reset() override;

  //! Load opacity from data file
  virtual void load();

  //! Get optical properties
  torch::Tensor forward(torch::Tensor var_x);
};
TORCH_MODULE(AbsorberRFM);

}  // namespace harp
