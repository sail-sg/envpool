#ifndef ENVPOOL_CLASSIC_CONTROL_RENDER_UTILS_H_
#define ENVPOOL_CLASSIC_CONTROL_RENDER_UTILS_H_

#include <cstdint>

namespace classic_control {
namespace rendering {

void RenderCartPole(double x, double theta, int width, int height,
                    std::uint8_t* rgb);

void RenderPendulum(double theta, bool has_last_u, double last_u, int width,
                    int height, std::uint8_t* rgb);

void RenderMountainCar(double pos, double goal_pos, int width, int height,
                       std::uint8_t* rgb);

void RenderAcrobot(double theta1, double theta2, int width, int height,
                   std::uint8_t* rgb);

}  // namespace rendering
}  // namespace classic_control

#endif  // ENVPOOL_CLASSIC_CONTROL_RENDER_UTILS_H_
