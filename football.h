#ifndef _GAME_ENV
#define _GAME_ENV

#define DO_VALIDATION ;
#define X_FIELD_SCALE 54.4
#define Y_FIELD_SCALE -83.6
#define Z_FIELD_SCALE 1
#define MAX_PLAYERS 11
#define CHECK(a) assert(a);


#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/random.hpp>
#include <boost/weak_ptr.hpp>
#include <iostream>
#include <algorithm>
#include <boost/assert.hpp>
#include <boost/config/workaround.hpp>
#include <boost/smart_ptr/detail/sp_convertible.hpp>
#include <boost/smart_ptr/detail/sp_nullptr_t.hpp>
#include <boost/smart_ptr/detail/sp_noexcept.hpp>
#include <deque>
#include <map>
#include <mutex>
#include <SDL_ttf.h>
#include <boost/thread/condition.hpp>

#if !defined(BOOST_NO_IOSTREAM)
#if !defined(BOOST_NO_IOSFWD)
#endif
#endif
#define SHARED_PTR boost::shared_ptr
#define WEAK_PTR boost::weak_ptr

constexpr real pi = 3.1415926535897932384626433832795028841972f; // last decimal rounded ;)
const float FORMATION_Y_SCALE = -2.36f;
const unsigned int ballPredictionSize_ms = 3000;
const unsigned int cachedPredictions = 100;
const unsigned int ballHistorySize = 401;
const float idleVelocity = 0.0f;
const float dribbleVelocity = 3.5f;
const float walkVelocity = 5.0f;
const float sprintVelocity = 8.0f;

const float animSprintVelocity = 7.0f;

const float idleDribbleSwitch = 1.8f;
const float dribbleWalkSwitch = 4.2f;
const float walkSprintSwitch = 6.0f;

typedef std::vector<std::string> StringVector;
typedef std::string screenshoot;
typedef float real;
typedef std::vector<Plane> vector_Planes;
typedef std::map <std::string, std::string> map_Properties;
typedef intrusive_ptr<Node> NodeMap[body_part_max];
typedef std::vector<MovementHistoryEntry> MovementHistory;

real clamp(const real value, const real min, const real max) {
    DO_VALIDATION;
    assert(max >= min);
    if (min > value) return min;
    if (max < value) return max;
    return value;
}

void Log(e_LogType logType, std::string className, std::string methodName, std::string message);
void intrusive_ptr_add_ref(RefCounted *p);
void intrusive_ptr_release(RefCounted *p);
float EnumToFloatVelocity(e_Velocity velocity) { DO_VALIDATION;
  switch (velocity) { DO_VALIDATION;
    case e_Velocity_Idle:
      return idleVelocity;
      break;
    case e_Velocity_Dribble:
      return dribbleVelocity;
      break;
    case e_Velocity_Walk:
      return walkVelocity;
      break;
    case e_Velocity_Sprint:
      return sprintVelocity;
      break;
  }
  return 0;
}

struct Position {
    Position(float x = 0.0, float y = 0.0, float z = 0.0, bool env_coords = false) {
        if (env_coords) {
            value[0] = X_FIELD_SCALE * x;
            value[1] = Y_FIELD_SCALE * y;
            value[2] = Z_FIELD_SCALE * z;
        } 
        else {
            value[0] = x;
            value[1] = y;
            value[2] = z;
        }
    }
    Position(const Position& other) {
        value[0] = other.value[0];
        value[1] = other.value[1];
        value[2] = other.value[2];
    }
    Position& operator=(float* position) {
        value[0] = position[0];
        value[1] = position[1];
        value[2] = position[2];
        return *this;
    }
    bool operator == (const Position& f) const {
        return value[0] == f.value[0] &&
            value[1] == f.value[1] &&
            value[2] == f.value[2];
    }
    float env_coord(int index) const;
    std::string debug();
private:
    float value[3];
};

enum e_SystemType {
    e_SystemType_None = 0,
    e_SystemType_Graphics = 1,
    e_SystemType_Physics = 2,
    e_SystemType_Audio = 3,
    e_SystemType_UserStart = 4
};

enum e_ObjectType {
    e_ObjectType_Camera = 1,
    e_ObjectType_Image2D = 2,
    e_ObjectType_Geometry = 3,
    e_ObjectType_Skybox = 4,
    e_ObjectType_Light = 5,
    e_ObjectType_UserStart = 7
  };

enum e_SceneType {
    e_SceneType_Scene2D = 1,
    e_SceneType_Scene3D = 2
};

enum e_PlayerRole {
    e_PlayerRole_GK,
    e_PlayerRole_CB,
    e_PlayerRole_LB,
    e_PlayerRole_RB,
    e_PlayerRole_DM,
    e_PlayerRole_CM,
    e_PlayerRole_LM,
    e_PlayerRole_RM,
    e_PlayerRole_AM,
    e_PlayerRole_CF,
};

enum e_CullingMode {
    e_CullingMode_Off,
    e_CullingMode_Front,
    e_CullingMode_Back
  };

enum e_Velocity {
  e_Velocity_Idle,
  e_Velocity_Dribble,
  e_Velocity_Walk,
  e_Velocity_Sprint
};

enum e_StrictMovement {
  e_StrictMovement_False,
  e_StrictMovement_True,
  e_StrictMovement_Dynamic
};

enum e_DefString {
  e_DefString_Empty = 0,
  e_DefString_OutgoingSpecialState = 1,
  e_DefString_IncomingSpecialState = 2,
  e_DefString_SpecialVar1 = 3,
  e_DefString_SpecialVar2 = 4,
  e_DefString_Type = 5,
  e_DefString_Trap = 6,
  e_DefString_Deflect = 7,
  e_DefString_Interfere = 8,
  e_DefString_Trip = 9,
  e_DefString_ShortPass = 10,
  e_DefString_LongPass = 11,
  e_DefString_Shot = 12,
  e_DefString_Sliding = 13,
  e_DefString_Movement = 14,
  e_DefString_Special = 15,
  e_DefString_BallControl = 16,
  e_DefString_HighPass = 17,
  e_DefString_Catch = 18,
  e_DefString_OutgoingRetainState = 19,
  e_DefString_IncomingRetainState = 20,
  e_DefString_Size = 21
};

enum BodyPart {
    middle,
    neck,
    left_thigh,
    right_thigh,
    left_knee,
    right_knee,
    left_ankle,
    right_ankle,
    left_shoulder,
    right_shoulder,
    left_elbow,
    right_elbow,
    body,
    player,
    body_part_max
  };

enum e_FunctionType {
  e_FunctionType_None,
  e_FunctionType_Movement,
  e_FunctionType_BallControl,
  e_FunctionType_Trap,
  e_FunctionType_ShortPass,
  e_FunctionType_LongPass,
  e_FunctionType_HighPass,
  e_FunctionType_Header,
  e_FunctionType_Shot,
  e_FunctionType_Deflect,
  e_FunctionType_Catch,
  e_FunctionType_Interfere,
  e_FunctionType_Trip,
  e_FunctionType_Sliding,
  e_FunctionType_Special
};

enum e_BlendingMode {
    e_BlendingMode_Off,
    e_BlendingMode_On
  };

enum e_BlendingFunction {
    e_BlendingFunction_Zero,
    e_BlendingFunction_One
};

enum PlayerStat {
  physical_balance,
  physical_reaction,
  physical_acceleration,
  physical_velocity,
  physical_stamina,
  physical_agility,
  physical_shotpower,
  technical_standingtackle,
  technical_slidingtackle,
  technical_ballcontrol,
  technical_dribble,
  technical_shortpass,
  technical_highpass,
  technical_header,
  technical_shot,
  technical_volley,
  mental_calmness,
  mental_workrate,
  mental_resilience,
  mental_defensivepositioning,
  mental_offensivepositioning,
  mental_vision,
  player_stat_max
};

enum e_DepthFunction {
  e_DepthFunction_Never,
  e_DepthFunction_Equal,
  e_DepthFunction_Greater,
  e_DepthFunction_Less,
  e_DepthFunction_GreaterOrEqual,
  e_DepthFunction_LessOrEqual,
  e_DepthFunction_NotEqual,
  e_DepthFunction_Always
};

enum e_TextureMode {
    e_TextureMode_Off,
    e_TextureMode_2D
};

enum e_InternalPixelFormat {
//    e_InternalPixelFormat_RGB,
    e_InternalPixelFormat_RGB8,
    e_InternalPixelFormat_SRGB8,
    e_InternalPixelFormat_RGB16,
//    e_InternalPixelFormat_RGBA,
    e_InternalPixelFormat_RGBA8,
    e_InternalPixelFormat_SRGBA8,
    e_InternalPixelFormat_RGBA16,
    e_InternalPixelFormat_RGBA16F,
    e_InternalPixelFormat_RGBA32F,

    e_InternalPixelFormat_RGBA4,
	e_InternalPixelFormat_RGB5_A1,
	e_InternalPixelFormat_DepthComponent,
	e_InternalPixelFormat_DepthComponent16,
	e_InternalPixelFormat_DepthComponent24,
	e_InternalPixelFormat_DepthComponent32,
	e_InternalPixelFormat_DepthComponent32F,
	e_InternalPixelFormat_StencilIndex8
  };

enum e_GameMode {
    e_GameMode_Normal,
    e_GameMode_KickOff,
    e_GameMode_GoalKick,
    e_GameMode_FreeKick,
    e_GameMode_Corner,
    e_GameMode_ThrowIn,
    e_GameMode_Penalty,
};

enum e_TextType {
    e_TextType_Default,
    e_TextType_DefaultOutline,
    e_TextType_Caption,
    e_TextType_Title,
    e_TextType_ToolTip
  };

enum e_OfficialType {
  e_OfficialType_Referee,
  e_OfficialType_Linesman
};

enum e_DecorationType {
    e_DecorationType_Dark1,
    e_DecorationType_Dark2,
    e_DecorationType_Bright1,
    e_DecorationType_Bright2,
    e_DecorationType_Toggled
  };

enum e_TouchType {
  e_TouchType_Intentional_Kicked, // goalies can't touch this
  e_TouchType_Intentional_Nonkicked, // headers and such
  e_TouchType_Accidental, // collisions
  e_TouchType_None,
  e_TouchType_SIZE
};

enum e_LogType {
    e_Warning,
    e_Error,
    e_FatalError
  };

enum e_InterruptAnim {
  e_InterruptAnim_None,
  e_InterruptAnim_Switch,
  e_InterruptAnim_Sliding,
  e_InterruptAnim_Bump,
  e_InterruptAnim_Trip,
  e_InterruptAnim_Cheat,
  e_InterruptAnim_Cancel,
  e_InterruptAnim_ReQueue
};

enum e_LocalMode {
    e_LocalMode_Relative,
    e_LocalMode_Absolute
  };

enum e_SpatialDataType {
    e_SpatialDataType_Position,
    e_SpatialDataType_Rotation,
    e_SpatialDataType_Both
};

enum e_Foot {
    e_Foot_Left,
    e_Foot_Right
  };

enum e_PlayerColor {
  e_PlayerColor_Blue,
  e_PlayerColor_Green,
  e_PlayerColor_Red,
  e_PlayerColor_Yellow,
  e_PlayerColor_Purple,
  e_PlayerColor_Default
};

enum e_ButtonFunction {
  e_ButtonFunction_LongPass,
  e_ButtonFunction_HighPass,
  e_ButtonFunction_ShortPass,
  e_ButtonFunction_Shot,
  e_ButtonFunction_KeeperRush,
  e_ButtonFunction_Sliding,
  e_ButtonFunction_Pressure,
  e_ButtonFunction_TeamPressure,
  e_ButtonFunction_Switch,
  e_ButtonFunction_Sprint,
  e_ButtonFunction_Dribble,
  e_ButtonFunction_Size
};

enum e_MatchPhase {
  e_MatchPhase_PreMatch,
  e_MatchPhase_1stHalf,
  e_MatchPhase_2ndHalf,
};

enum e_PixelFormat {
    e_PixelFormat_Alpha,
    e_PixelFormat_RGB,
    e_PixelFormat_RGBA,
    e_PixelFormat_DepthComponent,
    e_PixelFormat_Luminance
  };

enum e_ViewRenderTarget {
    e_ViewRenderTarget_Texture,
    e_ViewRenderTarget_Context
  };

enum e_TargetAttachment {
    e_TargetAttachment_None,
    e_TargetAttachment_Front,
    e_TargetAttachment_Back,
    e_TargetAttachment_Depth, // can not be used with drawbuffers
    e_TargetAttachment_Stencil, // can not be used with drawbuffers
    e_TargetAttachment_Color0,
    e_TargetAttachment_Color1,
    e_TargetAttachment_Color2,
    e_TargetAttachment_Color3
  };

enum e_DecayType {
  e_DecayType_Constant,
  e_DecayType_Variable
};

enum e_MagnetType {
  e_MagnetType_Attract,
  e_MagnetType_Repel
};

struct PlayerInfo {
    PlayerInfo() { }
    PlayerInfo(const PlayerInfo& f) {
        player_position = f.player_position;
        player_direction = f.player_direction;
        has_card = f.has_card;
        is_active = f.is_active;
        tired_factor = f.tired_factor;
        role = f.role;
        designated_player = f.designated_player;
    }
    bool operator == (const PlayerInfo& f) const {
        return player_position == f.player_position &&
            player_direction == f.player_direction &&
            has_card == f.has_card &&
            is_active == f.is_active &&
            tired_factor == f.tired_factor &&
            role == f.role &&
            designated_player == f.designated_player;
    }
    Position player_position;
    Position player_direction;
    bool has_card = false;
    bool is_active = true;
    bool designated_player = false;
    float tired_factor = 0.0f;
    e_PlayerRole role = e_PlayerRole_GK;
};

struct ControllerInfo {
    ControllerInfo() {}
    ControllerInfo(int controller_player) : controlled_player(controlled_player) {}
    bool operator == (const ControllerInfo& f)const {
        return controlled_player == f.controlled_player;
    }
    int controlled_player = -1;
};

struct sharedInfo {
    Position ball_position;
    Position ball_direction;
    Position ball_rotation;
    std::vector<PlayerInfo> left_team;
    std::vector<PlayerInfo> right_team;
    std::vector<ControllerInfo> left_controllers;
    std::vector<ControllerInfo> right_controllers;
    int left_goals, right_goals;
    e_GameMode game_mode;
    bool is_in_play = false;
    int ball_owned_team = 0;
    int ball_owned_player = 0;
    int step = 0;
};

class radian {
public:
    radian() {DO_VALIDATION;}
    radian(float r) : _angle(r) {DO_VALIDATION;}
    std::ostream& operator<<(std::ostream& os) {
        os << _angle << " " << _rotated;
        return os;
    }
    radian &operator+=(radian r) { DO_VALIDATION;
        _angle += r._angle;
        _rotated ^= r._rotated;
        return *this;
    }
    radian &operator-=(radian r){
        _angle -= r._angle;
        _rotated ^= r._rotated;
        return *this;
    }
    radian &operator/=(radian r) { DO_VALIDATION;
        *this = radian(real(*this) / real(r));
        return *this;
    }
    radian &operator*=(radian r) { DO_VALIDATION;
        *this = radian(real(*this) * real(r));
        return *this;
    }
    operator real() const {
        if (_rotated) { DO_VALIDATION;
            return _angle - pi;
        }
        return _angle;
    }
    void Mirror() { DO_VALIDATION;
        _rotated = !_rotated;
    }
private:
    float _angle = 0.0f;
    bool _rotated = false;
};

class Matrix4 {

    public:
      Matrix4();
      Matrix4(const real values[16]);
      virtual ~Matrix4();

      // ----- operator overloading
      void operator = (const Matrix3 &mat3);
      Matrix4 operator * (const Matrix4 &multiplier) const;
      bool operator == (const Matrix4 &mat);
      bool operator != (const Matrix4 &mat);

      // ----- mathematics
      Matrix4 GetInverse() const;
      void Transpose();
      Matrix4 GetTransposed() const;
      void SetTranslation(const Vector3 &trans);
      Vector3 GetTranslation() const;
      void Translate(const Vector3 &trans);
      Matrix4 GetTranslated(const Vector3 &trans);
      void SetScale(const Vector3 &scale);
      Vector3 GetScale() const;
      void Construct(const Vector3 &position, const Vector3 &scale, const Quaternion &rotation);
      void ConstructInverse(const Vector3 &position, const Vector3 &scale, const Quaternion &rotation);
      void MultiplyVec4(float x, float y, float z, float w, float &rx, float &ry, float &rz, float &rw);
      void ConstructProjection(float fov, float aspect, float zNear, float zFar);
      void ConstructOrtho(float left, float right, float bottom, float top, float zNear, float zFar);

      real elements[16];

    protected:

    private:

};

class Matrix3 {

    public:
      Matrix3();
      Matrix3(real values[9]);
      Matrix3(real v1, real v2, real v3, real v4, real v5, real v6, real v7, real v8, real v9);
      Matrix3(const Matrix3 &mat3);
      Matrix3(const Matrix4 &mat4);
      virtual ~Matrix3();

      // ----- operator overloading
      void operator = (const Matrix4 &mat4);
      Matrix3 operator * (const Matrix3 &multiplier) const;
      Vector3 operator * (const Vector3 &multiplier) const;

      // ----- mathematics
      void Transpose();

      static const Matrix3 ZERO;
      static const Matrix3 IDENTITY;
      real elements[9];

    protected:

    private:

};

class Quaternion{
public:
    inline Quaternion(){
        elements[0] = 0;
        elements[1] = 0;
        elements[2] = 0;
        elements[3] = 1;
    }
    Quaternion(real x, real y, real z, real w);
    Quaternion(real values[4]);
    void Set(real x, real y, real z, real w);
    void Set(const Quaternion &quat);
    void Set(const Matrix3 &mat);
    bool operator != (const Quaternion &fac) const;
    const Quaternion operator * (float scale) const;
    Vector3 operator * (const Vector3 &fac) const;
    void operator = (const Vector3 &vec);
    Quaternion operator * (const Quaternion &fac) const;
    Quaternion operator + (const Quaternion &q2) const;
    Quaternion operator - (const Quaternion &q2) const;
    Quaternion operator - () const;

    Quaternion GetInverse() const;
    void ConstructMatrix(Matrix3 &rotation) const;
    void GetAngles(real &X, real &Y, real &Z) const;
    void SetAngles(real X, real Y, real Z);
    void GetAngleAxis(radian &rfangle, Vector3 &rkaxis) const;
    void SetAngleAxis(const radian &rfangle, const Vector3 &rkaxis);
    void conjugate();
    Quaternion conjugate_get() const;
    void scale(const real fac);
    real GetMagnitude() const;
    void Normalize();
    Quaternion GetNormalized() const;
    float GetDotProduct(const Quaternion &subject) const;
    Quaternion GetLerped(float bias, const Quaternion &to) const; // bias > 1 can extrapolate small angles in a hacky, incorrect way *edit: is that so? i'm not so sure
    Quaternion GetSlerped(float bias, const Quaternion &to) const;
    Quaternion GetRotationTo(const Quaternion &to) const;
    Quaternion GetRotationMultipliedBy(float factor) const;
    radian GetRotationAngle(const Quaternion &to) const {
        return 2.0f * std::acos(clamp(this->GetDotProduct(to), -1.0f, 1.0f));
    }
    float MakeSameNeighborhood(const Quaternion &src); // returns dot product as added bonus! ;)
    real elements[4];
};

class Vector3 {
public:
    Vector3();
    Vector3(const Vector3 &src) {
        coords[0] = src.coords[0];
        coords[1] = src.coords[1];
        coords[2] = src.coords[2];
    }
    Vector3(real xyz);
    Vector3(real x, real y, real z);
    void Mirror() { coords[0] = -coords[0]; coords[1] = -coords[1]; }
    void Set(real xyz);
    void Set(real x, real y, real z);
    void Set(const Vector3 &vec);
    float GetEnvCoord(int index) const;
    void SetEnvCoord(int index, float value);
    bool operator == (const Vector3 &vector) const;
    bool operator != (const Vector3 &vector) const;
    void operator = (const Quaternion &quat);
    void operator = (const real src);
    void operator = (const Vector3 &src);
    Vector3 operator * (const real scalar) const;
    Vector3 operator * (const Vector3 &scalar) const;
    Vector3 &operator *= (const real scalar);
    Vector3 &operator *= (const Vector3 &scalar);
    Vector3 &operator *= (const Matrix3 &mat);
    Vector3 &operator *= (const Matrix4 &mat);
    Vector3 operator / (const real scalar) const;
    Vector3 operator / (const Vector3 &scalar) const;
    Vector3 &operator /= (const Vector3 &scalar);
    Vector3 &operator += (const real scalar);
    Vector3 &operator += (const Vector3 &scalar);
    Vector3 &operator -= (const Vector3 &scalar);
    Vector3 operator + (const Vector3 &vec) const;
    const Vector3 &operator + () const;
    Vector3 operator + (const real value) const;
    Vector3 operator - (const Vector3 &vec) const;
    Vector3 operator - () const;
    Vector3 operator - (const real value) const;
    bool operator < (const Vector3 &vector) const;
    Vector3 GetCrossProduct(const Vector3 &fac) const;
    real GetDotProduct(const Vector3 &fac) const;
    void ConstructMatrix(Matrix3 &mat) const;
    void FastNormalize();
    void Normalize(const Vector3 &ifNull);
    void Normalize();
    void NormalizeTo(float length);
    void NormalizeMax(float length);
    Vector3 GetNormalized(const Vector3 &ifNull) const;
    Vector3 GetNormalized() const;
    Vector3 GetNormalizedTo(float length) const;
    Vector3 GetNormalizedMax(float length) const;
    real GetDistance(const Vector3 &fac) const;
    real GetLength() const;
    real GetSquaredLength() const;
    radian GetAngle2D() const;
    radian GetAngle2D(const Vector3 &test) const;
    void Rotate(const Quaternion &quat);
    void Rotate2D(const radian angle);
    Vector3 GetRotated2D(const radian angle) const;
    Vector3 Get2D() const;
    bool Compare(const Vector3 &test) const;
    Vector3 GetAbsolute() const;
    Vector3 EnforceMaximumDeviation(const Vector3 &deviant, float maxDeviation) const;
    Vector3 GetClamped2D(const Vector3 &v1, const Vector3 &v2) const;
    void Extrapolate(const Vector3 &direction, unsigned long time);
    real coords[3];
};

class RefCounted {

    public:
      RefCounted();
      virtual ~RefCounted();

      RefCounted(const RefCounted &src);
      RefCounted& operator=(const RefCounted &src);

      unsigned long GetRefCount();

    protected:

    private:
      volatile long refCount = 0;
      friend void intrusive_ptr_add_ref(RefCounted *p);
      friend void intrusive_ptr_release(RefCounted *p);

  };

class Observer : public RefCounted {

    public:
      Observer();
      virtual ~Observer();

      void SetSubjectPtr(void *subjectPtr);

    protected:
      void *subjectPtr;

  };

class Interpreter : public Observer {

    public:
      virtual e_SystemType GetSystemType() const = 0;
      virtual void OnSynchronize() { DO_VALIDATION;
        Log(e_FatalError, "Interpreter", "OnSynchronize", "OnSynchronize not written yet for this object! N00B!");
      }

    protected:

};

template<class T> class intrusive_ptr
{
private:

    typedef intrusive_ptr this_type;

public:

    typedef T element_type;

    BOOST_CONSTEXPR intrusive_ptr() BOOST_SP_NOEXCEPT : px( 0 )
    {
    }

    intrusive_ptr( T * p, bool add_ref = true ): px( p )
    {
        if( px != 0 && add_ref ) intrusive_ptr_add_ref( px );
    }

#if !defined(BOOST_NO_MEMBER_TEMPLATES) || defined(BOOST_MSVC6_MEMBER_TEMPLATES)

    template<class U>
#if !defined( BOOST_SP_NO_SP_CONVERTIBLE )

    intrusive_ptr( intrusive_ptr<U> const & rhs, typename boost::detail::sp_enable_if_convertible<U,T>::type = boost::detail::sp_empty() )

#else

    intrusive_ptr( intrusive_ptr<U> const & rhs )

#endif
    : px( rhs.get() )
    {
        if( px != 0 ) intrusive_ptr_add_ref( px );
    }

#endif

    intrusive_ptr(intrusive_ptr const & rhs): px( rhs.px )
    {
        if( px != 0 ) intrusive_ptr_add_ref( px );
    }

    ~intrusive_ptr()
    {
        if( px != 0 ) intrusive_ptr_release( px );
    }

#if !defined(BOOST_NO_MEMBER_TEMPLATES) || defined(BOOST_MSVC6_MEMBER_TEMPLATES)

    template<class U> intrusive_ptr & operator=(intrusive_ptr<U> const & rhs)
    {
        this_type(rhs).swap(*this);
        return *this;
    }

#endif

// Move support

#if !defined( BOOST_NO_CXX11_RVALUE_REFERENCES )

    intrusive_ptr(intrusive_ptr && rhs) BOOST_SP_NOEXCEPT : px( rhs.px )
    {
        rhs.px = 0;
    }

    intrusive_ptr & operator=(intrusive_ptr && rhs) BOOST_SP_NOEXCEPT
    {
        this_type( static_cast< intrusive_ptr && >( rhs ) ).swap(*this);
        return *this;
    }

    template<class U> friend class intrusive_ptr;

    template<class U>
#if !defined( BOOST_SP_NO_SP_CONVERTIBLE )

    intrusive_ptr(intrusive_ptr<U> && rhs, typename boost::detail::sp_enable_if_convertible<U,T>::type = boost::detail::sp_empty())

#else

    intrusive_ptr(intrusive_ptr<U> && rhs)

#endif        
    : px( rhs.px )
    {
        rhs.px = 0;
    }

    template<class U>
    intrusive_ptr & operator=(intrusive_ptr<U> && rhs) BOOST_SP_NOEXCEPT
    {
        this_type( static_cast< intrusive_ptr<U> && >( rhs ) ).swap(*this);
        return *this;
    }

#endif

    intrusive_ptr & operator=(intrusive_ptr const & rhs)
    {
        this_type(rhs).swap(*this);
        return *this;
    }

    intrusive_ptr & operator=(T * rhs)
    {
        this_type(rhs).swap(*this);
        return *this;
    }

    void reset()
    {
        this_type().swap( *this );
    }

    void reset( T * rhs )
    {
        this_type( rhs ).swap( *this );
    }

    void reset( T * rhs, bool add_ref )
    {
        this_type( rhs, add_ref ).swap( *this );
    }

    T * get() const BOOST_SP_NOEXCEPT
    {
        return px;
    }

    T * detach() BOOST_SP_NOEXCEPT
    {
        T * ret = px;
        px = 0;
        return ret;
    }

    T & operator*() const BOOST_SP_NOEXCEPT_WITH_ASSERT
    {
        BOOST_ASSERT( px != 0 );
        return *px;
    }

    T * operator->() const BOOST_SP_NOEXCEPT_WITH_ASSERT
    {
        BOOST_ASSERT( px != 0 );
        return px;
    }

// implicit conversion to "bool"
#include <boost/smart_ptr/detail/operator_bool.hpp>

    void swap(intrusive_ptr & rhs) BOOST_SP_NOEXCEPT
    {
        T * tmp = px;
        px = rhs.px;
        rhs.px = tmp;
    }

private:

    T * px;
};

template<class T, class U> inline bool operator==(intrusive_ptr<T> const & a, intrusive_ptr<U> const & b) BOOST_SP_NOEXCEPT
{
    return a.get() == b.get();
}

template<class T, class U> inline bool operator!=(intrusive_ptr<T> const & a, intrusive_ptr<U> const & b) BOOST_SP_NOEXCEPT
{
    return a.get() != b.get();
}

template<class T, class U> inline bool operator==(intrusive_ptr<T> const & a, U * b) BOOST_SP_NOEXCEPT
{
    return a.get() == b;
}

template<class T, class U> inline bool operator!=(intrusive_ptr<T> const & a, U * b) BOOST_SP_NOEXCEPT
{
    return a.get() != b;
}

template<class T, class U> inline bool operator==(T * a, intrusive_ptr<U> const & b) BOOST_SP_NOEXCEPT
{
    return a == b.get();
}

template<class T, class U> inline bool operator!=(T * a, intrusive_ptr<U> const & b) BOOST_SP_NOEXCEPT
{
    return a != b.get();
}

#if __GNUC__ == 2 && __GNUC_MINOR__ <= 96

// Resolve the ambiguity between our op!= and the one in rel_ops

template<class T> inline bool operator!=(intrusive_ptr<T> const & a, intrusive_ptr<T> const & b) BOOST_SP_NOEXCEPT
{
    return a.get() != b.get();
}

#endif

#if !defined( BOOST_NO_CXX11_NULLPTR )

template<class T> inline bool operator==( intrusive_ptr<T> const & p, boost::detail::sp_nullptr_t ) BOOST_SP_NOEXCEPT
{
    return p.get() == 0;
}

template<class T> inline bool operator==( boost::detail::sp_nullptr_t, intrusive_ptr<T> const & p ) BOOST_SP_NOEXCEPT
{
    return p.get() == 0;
}

template<class T> inline bool operator!=( intrusive_ptr<T> const & p, boost::detail::sp_nullptr_t ) BOOST_SP_NOEXCEPT
{
    return p.get() != 0;
}

template<class T> inline bool operator!=( boost::detail::sp_nullptr_t, intrusive_ptr<T> const & p ) BOOST_SP_NOEXCEPT
{
    return p.get() != 0;
}

#endif

template<class T> inline bool operator<(intrusive_ptr<T> const & a, intrusive_ptr<T> const & b) BOOST_SP_NOEXCEPT
{
    return std::less<T *>()(a.get(), b.get());
}

template<class T> void swap(intrusive_ptr<T> & lhs, intrusive_ptr<T> & rhs) BOOST_SP_NOEXCEPT
{
    lhs.swap(rhs);
}

// mem_fn support

template<class T> T * get_pointer(intrusive_ptr<T> const & p) BOOST_SP_NOEXCEPT
{
    return p.get();
}

// pointer casts

template<class T, class U> intrusive_ptr<T> static_pointer_cast(intrusive_ptr<U> const & p)
{
    return static_cast<T *>(p.get());
}

template<class T, class U> intrusive_ptr<T> const_pointer_cast(intrusive_ptr<U> const & p)
{
    return const_cast<T *>(p.get());
}

template<class T, class U> intrusive_ptr<T> dynamic_pointer_cast(intrusive_ptr<U> const & p)
{
    return dynamic_cast<T *>(p.get());
}

#if !defined( BOOST_NO_CXX11_RVALUE_REFERENCES )

template<class T, class U> intrusive_ptr<T> static_pointer_cast( intrusive_ptr<U> && p ) BOOST_SP_NOEXCEPT
{
    return intrusive_ptr<T>( static_cast<T*>( p.detach() ), false );
}

template<class T, class U> intrusive_ptr<T> const_pointer_cast( intrusive_ptr<U> && p ) BOOST_SP_NOEXCEPT
{
    return intrusive_ptr<T>( const_cast<T*>( p.detach() ), false );
}

template<class T, class U> intrusive_ptr<T> dynamic_pointer_cast( intrusive_ptr<U> && p ) BOOST_SP_NOEXCEPT
{
    T * p2 = dynamic_cast<T*>( p.get() );

    intrusive_ptr<T> r( p2, false );

    if( p2 ) p.detach();

    return r;
}

#endif // defined( BOOST_NO_CXX11_RVALUE_REFERENCES )

#if !defined(BOOST_NO_IOSTREAM)

#if defined(BOOST_NO_TEMPLATED_IOSTREAMS) || ( defined(__GNUC__) &&  (__GNUC__ < 3) )

template<class Y> std::ostream & operator<< (std::ostream & os, intrusive_ptr<Y> const & p)
{
    os << p.get();
    return os;
}

#else

// in STLport's no-iostreams mode no iostream symbols can be used
#ifndef _STLP_NO_IOSTREAMS

# if defined(BOOST_MSVC) && BOOST_WORKAROUND(BOOST_MSVC, < 1300 && __SGI_STL_PORT)
// MSVC6 has problems finding std::basic_ostream through the using declaration in namespace _STL
using std::basic_ostream;
template<class E, class T, class Y> basic_ostream<E, T> & operator<< (basic_ostream<E, T> & os, intrusive_ptr<Y> const & p)
# else
template<class E, class T, class Y> std::basic_ostream<E, T> & operator<< (std::basic_ostream<E, T> & os, intrusive_ptr<Y> const & p)
# endif 
{
    os << p.get();
    return os;
}

#endif // _STLP_NO_IOSTREAMS

#endif // __GNUC__ < 3

#endif // !defined(BOOST_NO_IOSTREAM)

template< class T > struct hash;

template< class T > std::size_t hash_value( intrusive_ptr<T> const & p ) BOOST_SP_NOEXCEPT
{
    return boost::hash< T* >()( p.get() );
}

#if !defined(BOOST_NO_CXX11_HDR_FUNCTIONAL)

namespace std
{

template<class T> struct hash< ::intrusive_ptr<T> >
{
    std::size_t operator()( ::intrusive_ptr<T> const & p ) const BOOST_SP_NOEXCEPT
    {
        return std::hash< T* >()( p.get() );
    }
};

} // namespace std

#endif // #if !defined(BOOST_NO_CXX11_HDR_FUNCTIONAL)
#endif  // #ifndef BOOST_SMART_PTR_INTRUSIVE_PTR_HPP_INCLUDED

struct MustUpdateSpatialData {
    bool haveTo = false;
    bool wihfi = true;
    e_SystemType excludeSystem;
};

template <class T = Observer>
class Subject {

    public:
      Subject() { DO_VALIDATION;
      }
      
      virtual ~Subject() { DO_VALIDATION;
        observers.clear();
      }

      virtual void Attach(intrusive_ptr<T> observer, void *thisPtr = 0) { DO_VALIDATION;

        observer->SetSubjectPtr(thisPtr);

        observers.push_back(observer);
      }

      virtual void Detach(intrusive_ptr<T> observer) { DO_VALIDATION;
        typename std::vector <intrusive_ptr<T> >::iterator o_iter = observers.begin();
        while (o_iter != observers.end()) { DO_VALIDATION;
          if ((*o_iter).get() == observer.get()) { DO_VALIDATION;
            (*o_iter).reset();
            o_iter = observers.erase(o_iter);
          } else {
            o_iter++;
          }
        }
      }

      virtual void DetachAll() { DO_VALIDATION;
        typename std::vector <intrusive_ptr<T> >::iterator o_iter = observers.begin();
        while (o_iter != observers.end()) { DO_VALIDATION;
          (*o_iter).reset();
          o_iter = observers.erase(o_iter);
        }
      }

    protected:
      std::vector <intrusive_ptr<T> > observers;

  };

class Plane {
    // vector 0 = a position on the plane
    // vector 1 = plane normal

    public:
      Plane();
      Plane(const Vector3 vec1, const Vector3 vec2);
      ~Plane();

      void Set(const Vector3 &pos, const Vector3 &dir);
      void SetVertex(unsigned char pos, const Vector3 &vec);
      const Vector3 &GetVertex(unsigned char pos) const;

      void CalculateDeterminant() const;
      real GetDeterminant() const;

    protected:
      Vector3 vertices[2];
      mutable real determinant;
      mutable bool _dirty_determinant = false;

    private:

  };

class AABB {

    public:
      AABB();
      AABB(const AABB &src);
      virtual ~AABB();

      AABB operator += (const AABB &add);
      AABB operator + (const Vector3 &vec) const;
      AABB operator * (const Quaternion &rot) const;

      void Reset();

      void SetMinXYZ(const Vector3 &min);
      void SetMaxXYZ(const Vector3 &max);

      real GetRadius() const;
      void GetCenter(Vector3 &center) const;
      bool Intersects(const Vector3 &center, const real radius) const;
      bool Intersects(const vector_Planes &planes) const;
      bool Intersects(const AABB &src) const;

      void MakeDirty() { DO_VALIDATION; radius_needupdate = true; center_needupdate = true; }
      Vector3 minxyz;
      Vector3 maxxyz;
      mutable real radius = 0.0f;
      mutable Vector3 center;

    protected:
      mutable bool radius_needupdate = false;
      mutable bool center_needupdate = false;

  };



class Spatial : public RefCounted {

    public:
      Spatial(const std::string &name);
      virtual ~Spatial();

      Spatial(const Spatial &src);

      virtual void Exit() = 0;

      void SetLocalMode(e_LocalMode localMode);
      e_LocalMode GetLocalMode();

      void SetName(const std::string &name);
      virtual const std::string GetName() const;

      void SetParent(Node *parent);

      void SetPosition(const Vector3 &newPosition, bool updateSpatialData = true);
      Vector3 GetPosition() const;

      virtual void SetRotation(const Quaternion &newRotation, bool updateSpatialData = true);
      virtual Quaternion GetRotation() const;

      virtual void SetScale(const Vector3 &newScale);
      virtual Vector3 GetScale() const;

      virtual Vector3 GetDerivedPosition() const;
      virtual Quaternion GetDerivedRotation() const;
      virtual Vector3 GetDerivedScale() const;

      virtual void RecursiveUpdateSpatialData(e_SpatialDataType spatialDataType, e_SystemType excludeSystem = e_SystemType_None) = 0;

      virtual void InvalidateBoundingVolume();
      virtual void InvalidateSpatialData();

      virtual AABB GetAABB() const;

    protected:
      std::string name;

      Node *parent;

      Vector3 position;
      Quaternion rotation;
      Vector3 scale;

      // cache
      mutable bool _dirty_DerivedPosition = false;
      mutable bool _dirty_DerivedRotation = false;
      mutable bool _dirty_DerivedScale = false;
      mutable Vector3 _cache_DerivedPosition;
      mutable Quaternion _cache_DerivedRotation;
      mutable Vector3 _cache_DerivedScale;

      e_LocalMode localMode;

      mutable AABBCache aabb;

  };

struct AABBCache {
    bool dirty = false;
    AABB aabb;
};

class Node : public Spatial {

    public:
      Node(const std::string &name);
      Node(const Node &source, const std::string &postfix, boost::shared_ptr<Scene3D> scene3D);
      virtual ~Node();

      virtual void Exit();

      void AddNode(intrusive_ptr<Node> node);
      void DeleteNode(intrusive_ptr<Node> node);
      void GetNodes(std::vector<intrusive_ptr<Node> > &gatherNodes,
                    bool recurse = false) const;

      void AddObject(intrusive_ptr<Object> object);
      intrusive_ptr<Object> GetObject(const std::string &name);
      void DeleteObject(intrusive_ptr<Object> object,
                        bool exitObject = true);

      void GetObjects(std::list<intrusive_ptr<Object> > &gatherObjects,
                      bool recurse = true, int depth = 0) const;

      void GetObjects(std::deque < intrusive_ptr<Object> > &gatherObjects, const vector_Planes &bounding, bool recurse = true, int depth = 0) const;
      void ProcessState(EnvState* state);

      template <class T>
      inline void GetObjects(e_ObjectType targetObjectType, std::list < intrusive_ptr<T> > &gatherObjects, bool recurse = true, int depth = 0) const {
        //objects.Lock();
        int objectsSize = objects.size();
        for (int i = 0; i < objectsSize; i++) { DO_VALIDATION;
          if (objects[i]->GetObjectType() == targetObjectType) { DO_VALIDATION;
            gatherObjects.push_back(boost::static_pointer_cast<T>(objects[i]));
          }
        }

        if (recurse) { DO_VALIDATION;
          int nodesSize = nodes.size();
          for (int i = 0; i < nodesSize; i++) { DO_VALIDATION;
            nodes[i]->GetObjects<T>(targetObjectType, gatherObjects, recurse, depth + 1);
          }
        }
      }

      template <class T>
      inline void GetObjects(e_ObjectType targetObjectType, std::deque < intrusive_ptr<T> > &gatherObjects, const vector_Planes &bounding, bool recurse = true, int depth = 0) const {
        int objectsSize = objects.size();
        for (int i = 0; i < objectsSize; i++) { DO_VALIDATION;
          if (objects[i]->GetObjectType() == targetObjectType) { DO_VALIDATION;
            if (objects[i]->GetAABB().Intersects(bounding)) gatherObjects.push_back(boost::static_pointer_cast<T>(objects[i]));
          }
        }

        if (recurse) { DO_VALIDATION;
          int nodesSize = nodes.size();
          for (int i = 0; i < nodesSize; i++) { DO_VALIDATION;
            if (nodes[i]->GetAABB().Intersects(bounding)) nodes[i]->GetObjects<T>(targetObjectType, gatherObjects, bounding, recurse, depth + 1);
          }
        }
      }

      void PokeObjects(e_ObjectType targetObjectType, e_SystemType targetSystem);

      virtual AABB GetAABB() const;

      virtual void RecursiveUpdateSpatialData(e_SpatialDataType spatialDataType, e_SystemType excludeSystem = e_SystemType_None);

    protected:
      mutable std::vector < intrusive_ptr<Node> > nodes;
      mutable std::vector < intrusive_ptr<Object> > objects;
  };
  
class Properties {

    public:
      Properties();
      Properties(std::vector<std::pair<std::string, float>> values);
      virtual ~Properties();

      bool Exists(const std::string &name) const;

      void Set(const std::string &name, const std::string &value);
      void SetInt(const std::string &name, int value);
      void Set(const std::string &name, real value);
      void SetBool(const std::string &name, bool value);
      const std::string &Get(
          const std::string &name,
          const std::string &defaultValue = "") const;
      bool GetBool(const std::string &name, bool defaultValue = false) const;
      real GetReal(const std::string &name, real defaultValue = 0) const;
      int GetInt(const std::string &name, int defaultValue = 0) const;
      void AddProperties(const Properties *userprops);
      void AddProperties(const Properties &userprops);
      const map_Properties *GetProperties() const;
      void ProcessState(EnvState* state);

     protected:
      map_Properties properties;
  };

class Object : public Subject<Interpreter>, public Spatial {

    public:
      Object(std::string name, e_ObjectType objectType);
      virtual ~Object();

      Object(const Object &src);

      virtual void Exit(); // ATOMIC

      virtual e_ObjectType GetObjectType();

      virtual bool IsEnabled() { DO_VALIDATION; return enabled; }
      virtual void Enable() { DO_VALIDATION; enabled = true; }
      virtual void Disable() { DO_VALIDATION; enabled = false; }

      virtual const Properties &GetProperties() const;
      virtual bool PropertyExists(const char *property) const;
      virtual const std::string &GetProperty(const char *property) const;

      virtual void SetProperties(Properties properties);
      virtual void SetProperty(const char *name, const char *value);

      virtual bool RequestPropertyExists(const char *property) const;
      virtual std::string GetRequestProperty(const char *property) const;
      virtual void AddRequestProperty(const char *property);
      virtual void SetRequestProperty(const char *property, const char *value);

      virtual void Synchronize();
      virtual void Poke(e_SystemType targetSystemType);

      virtual void RecursiveUpdateSpatialData(e_SpatialDataType spatialDataType, e_SystemType excludeSystem = e_SystemType_None);

      MustUpdateSpatialData updateSpatialDataAfterPoke;

      virtual intrusive_ptr<Interpreter> GetInterpreter(e_SystemType targetSystemType);

      virtual void SetPokePriority(int prio) { DO_VALIDATION; pokePriority = prio; }
      virtual int GetPokePriority() const { return pokePriority; }

      // set these before creating system objects

      Properties properties;



    protected:
      e_ObjectType objectType;

      mutable int pokePriority;

      // request these to be set by observing objects
      mutable Properties requestProperties;

      bool enabled = false;
      
  };

class ISystemObject {
    public:
      virtual ~ISystemObject() { DO_VALIDATION;};

      virtual intrusive_ptr<Interpreter> GetInterpreter(e_ObjectType objectType) = 0;

    protected:

};

class ISceneInterpreter : public Interpreter {

    public:
      virtual void OnLoad() = 0;
      virtual void OnUnload() = 0;

      virtual ISystemObject *CreateSystemObject(Object* object) = 0;

    protected:

};

class GraphicsScene {

    public:
      GraphicsScene(GraphicsSystem *graphicsSystem);
      virtual ~GraphicsScene();

      virtual GraphicsSystem *GetGraphicsSystem();

      virtual ISystemObject *CreateSystemObject(Object* object);

      virtual intrusive_ptr<ISceneInterpreter> GetInterpreter(e_SceneType sceneType);

    protected:
      GraphicsSystem *graphicsSystem;
};

class IScene : public Subject<ISceneInterpreter> {

    public:
      virtual void Init() = 0;
      virtual void Exit() = 0; // ATOMIC

      virtual void CreateSystemObjects(intrusive_ptr<Object> object) = 0;

      virtual void PokeObjects(e_ObjectType targetObjectType, e_SystemType targetSystemType) = 0;
      virtual bool SupportedObjectType(e_ObjectType objectType) const = 0;

    protected:

  };

template <typename T>
class Resource : public RefCounted {

    public:
      Resource(std::string identString) : resource(0), identString(identString) { DO_VALIDATION;
        resource = new T();
      }

      virtual ~Resource() { DO_VALIDATION;
        delete resource;
        resource = 0;
      }

      Resource(const Resource &src, const std::string &identString) : identString(identString) { DO_VALIDATION;
        this->resource = new T(*src.resource);
      }



      T *GetResource() { DO_VALIDATION;
        return resource;
      }

      std::string GetIdentString() { DO_VALIDATION;
        return identString;
      }

      T *resource;

    protected:
      const std::string identString;

  };

struct SDL_Surface;

class Surface {

    public:
      Surface();
      virtual ~Surface();
      Surface(const Surface &src);

      SDL_Surface *GetData();
      void SetData(SDL_Surface *surface);

      void Resize(int x, int y);  // 0 == dependent on other coord

    protected:
      SDL_Surface *surface;

  };

struct Material {
    intrusive_ptr < Resource<Surface> > diffuseTexture;
    intrusive_ptr < Resource<Surface> > normalTexture;
    intrusive_ptr < Resource<Surface> > specularTexture;
    intrusive_ptr < Resource<Surface> > illuminationTexture;
    float shininess = 0.0f;
    float specular_amount = 0.0f;
    Vector3 self_illumination;
  };

struct MaterializedTriangleMesh {
    Material material;

    float *vertices; // was: triangleMesh
    int verticesDataSize = 0; // was: triangleMeshSize

    /* contents:
    float vertices[verticesDataSize * 3];
    float normals[verticesDataSize * 3];
    float texturevertices[verticesDataSize * 3];
    float tangents[verticesDataSize * 3];
    float bitangents[verticesDataSize * 3];
    */

    std::vector<unsigned int> indices;
  };

class GeometryData {

    public:
      GeometryData();
      virtual ~GeometryData();
      GeometryData(const GeometryData &src);


      void AddTriangleMesh(Material material, float *vertices,
                           int verticesDataSize,
                           std::vector<unsigned int> indices);
      std::vector < MaterializedTriangleMesh > GetTriangleMeshes();
      std::vector < MaterializedTriangleMesh > &GetTriangleMeshesRef();
      void SetDynamic(bool dynamic) { DO_VALIDATION; isDynamic = dynamic; }
      bool IsDynamic() { DO_VALIDATION; return isDynamic; }

      AABB GetAABB() const;

    protected:
      bool isDynamic = false;
      std::vector < MaterializedTriangleMesh > triangleMeshes;

      mutable AABBCache aabb;

  };

class Geometry : public Object {

    public:
      Geometry(std::string name, e_ObjectType objectType = e_ObjectType_Geometry);
      Geometry(const Geometry &src, const std::string &postfix);
      virtual ~Geometry();

      virtual void Exit();

      void SetGeometryData(intrusive_ptr < Resource<GeometryData> > geometryData);
      intrusive_ptr < Resource<GeometryData> > GetGeometryData();

      void OnUpdateGeometryData(bool updateMaterials = true);

      virtual void Poke(e_SystemType targetSystemType);

      void RecursiveUpdateSpatialData(e_SpatialDataType spatialDataType, e_SystemType excludeSystem = e_SystemType_None);

      virtual AABB GetAABB() const;

    protected:
      intrusive_ptr < Resource<GeometryData> > geometryData;

  };

enum e_LightType {
    e_LightType_Directional,
    e_LightType_Point
  };

class Light : public Object {

    public:
      Light(std::string name);
      virtual ~Light();

      virtual void Exit();

      virtual void SetColor(const Vector3 &color);
      virtual Vector3 GetColor() const;

      virtual void SetRadius(float radius);
      virtual float GetRadius() const;

      virtual void SetType(e_LightType lightType);
      virtual e_LightType GetType() const;

      virtual void SetShadow(bool shadow);
      virtual bool GetShadow() const;

      virtual void UpdateValues();

      virtual void EnqueueShadowMap(intrusive_ptr<Camera> camera, std::deque < intrusive_ptr<Geometry> > visibleGeometry);
      virtual void Poke(e_SystemType targetSystemType);

      virtual void RecursiveUpdateSpatialData(e_SpatialDataType spatialDataType, e_SystemType excludeSystem = e_SystemType_None);

      virtual AABB GetAABB() const;

    protected:
      Vector3 color;
      float radius = 0.0f;
      e_LightType lightType;
      bool shadow = false;

  };

class Skybox : public Geometry {

    public:
      Skybox(std::string name);
      virtual ~Skybox();

    protected:

  };

class Camera : public Object {

    public:
      Camera(std::string name);
      virtual ~Camera();

      virtual void Init();
      virtual void Exit();

      virtual void SetFOV(float fov);
      virtual float GetFOV() const { return fov; }
      virtual void SetCapping(float nearCap, float farCap);
      virtual void GetCapping(float &nearCap, float &farCap) const { nearCap = this->nearCap; farCap = this->farCap; }


      virtual void EnqueueView(std::deque <intrusive_ptr<Geometry> > &visibleGeometry, std::deque < intrusive_ptr<Light> > &visibleLights, std::deque < intrusive_ptr<Skybox> > &skyboxes);
      virtual void Poke(e_SystemType targetSystemType);

      virtual void RecursiveUpdateSpatialData(e_SpatialDataType spatialDataType, e_SystemType excludeSystem = e_SystemType_None);

    protected:
      float fov = 0.0f;
      float nearCap = 0.0f;
      float farCap = 0.0f;

  };

class GraphicsTask {

    public:
      GraphicsTask(GraphicsSystem *system);
      ~GraphicsTask();
      void Render(bool swap_buffer);
    protected:
      bool Execute(intrusive_ptr<Camera> camera);
      void EnqueueShadowMap(intrusive_ptr<Camera> camera, intrusive_ptr<Light> light);

      GraphicsSystem *graphicsSystem;
  };

struct Shader {
    std::string name;
    unsigned int programID = 0;
    unsigned int vertexShaderID = 0;
    unsigned int fragmentShaderID = 0;
  };

class Texture {

    public:
      Texture();
      virtual ~Texture();

      void SetRenderer3D(Renderer3D *renderer3D);

      void DeleteTexture();
      int CreateTexture(e_InternalPixelFormat internalPixelFormat, e_PixelFormat pixelFormat, int width, int height, bool alpha, bool repeat, bool mipmaps, bool filter, bool compareDepth = false);
      void ResizeTexture(SDL_Surface *image, e_InternalPixelFormat internalPixelFormat, e_PixelFormat pixelFormat, bool alpha, bool mipmaps);
      void UpdateTexture(SDL_Surface *image, bool alpha, bool mipmaps);

      int GetID();

      void GetSize(int &width, int &height) const { width = this->width; height = this->height; }

    protected:
      int textureID = 0;
      Renderer3D *renderer3D;
      int width, height;

  };

struct Overlay2DQueueEntry {
    intrusive_ptr < Resource<Texture> > texture;
    int position[2];
    int size[2];
  };

struct LightQueueEntry {
    bool hasShadow = false;
    Matrix4 lightProjectionMatrix;
    Matrix4 lightViewMatrix;
    intrusive_ptr < Resource<Texture> > shadowMapTexture;
    Vector3 position;
    int type = 0; // 0 == directional, 1 == point
    Vector3 color;
    float radius = 0.0f;
    bool shadow = false;
    AABB aabb;
  };

struct View {
    e_ViewRenderTarget target;
    int targetTexID = 0;
    int x, y, width, height;

    int gBufferID = 0;
    int gBuffer_DepthTexID = 0;
    int gBuffer_AlbedoTexID = 0;
    int gBuffer_NormalTexID = 0;
    int gBuffer_AuxTexID = 0;

    int accumBufferID = 0;
    int accumBuffer_AccumTexID = 0;
    int accumBuffer_ModifierTexID = 0;
  };

struct VertexBufferID {
    VertexBufferID() { DO_VALIDATION;
      bufferID = -1;
    }
    int bufferID = 0; // -1 if uninitialized
    unsigned int vertexArrayID = 0;
    unsigned int elementArrayID = 0;
};

struct Renderer3DMaterial {
    intrusive_ptr < Resource<Texture> > diffuseTexture;
    intrusive_ptr < Resource<Texture> > normalTexture;
    intrusive_ptr < Resource<Texture> > specularTexture;
    intrusive_ptr < Resource<Texture> > illuminationTexture;
    float shininess = 0.0f;
    float specular_amount = 0.0f;
    Vector3 self_illumination;
  };

struct VertexBufferIndex {
    int startIndex = 0;
    int size = 0;
    Renderer3DMaterial material;
  };

class VertexBuffer {

    public:
      VertexBuffer();
      virtual ~VertexBuffer();

      void SetTriangleMesh(const std::vector<float>& vertices, unsigned int verticesDataSize, std::vector<unsigned int> indices);
      void TriangleMeshWasUpdatedExternally(unsigned int verticesDataSize, std::vector<unsigned int> indices);
      VertexBufferID CreateOrUpdateVertexBuffer(Renderer3D *renderer3D, bool dynamicBuffer);

      float* GetTriangleMesh();

      int GetVaoID();

      int GetVerticesDataSize();

     protected:
      std::vector<float> vertices;
      int verticesDataSize = 0;
      std::vector<unsigned int> indices;
      VertexBufferID vertexBufferID;
      int vertexCount = 0;
      Renderer3D *renderer3D;
      bool dynamicBuffer = false;

      bool sizeChanged = false;

  };

struct VertexBufferQueueEntry {
    std::list<VertexBufferIndex>* vertexBufferIndices;
    intrusive_ptr < Resource<VertexBuffer> > vertexBuffer;
    AABB aabb;
    Vector3 position;
    Quaternion rotation;

  };

enum e_VertexBufferUsage {
    e_VertexBufferUsage_StreamDraw,
    e_VertexBufferUsage_StreamRead,
    e_VertexBufferUsage_StreamCopy,
    e_VertexBufferUsage_StaticDraw,
    e_VertexBufferUsage_StaticRead,
    e_VertexBufferUsage_StaticCopy,
    e_VertexBufferUsage_DynamicDraw,
    e_VertexBufferUsage_DynamicRead,
    e_VertexBufferUsage_DynamicCopy
  };

enum e_RenderMode {
    e_RenderMode_GeometryOnly,
    e_RenderMode_Diffuse,
    e_RenderMode_Full
  };

class Renderer3D {

    public:
      virtual ~Renderer3D() { DO_VALIDATION;};
      virtual void SetContext() = 0;
      virtual void DisableContext() = 0;
      virtual const screenshoot& GetScreen() = 0;

      virtual void SwapBuffers() = 0;

      virtual void SetMatrix(const std::string &shaderUniformName, const Matrix4 &matrix) = 0;

      virtual void RenderOverlay2D(const std::vector<Overlay2DQueueEntry> &overlay2DQueue) = 0;
      virtual void RenderOverlay2D() = 0;
      virtual void RenderLights(std::deque<LightQueueEntry> &lightQueue, const Matrix4 &projectionMatrix, const Matrix4 &viewMatrix) = 0;


      // --- new & improved

      // init & exit
      virtual bool CreateContext(int width, int height, int bpp, bool fullscreen) = 0;
      virtual void Exit() = 0;

      virtual int CreateView(float x_percent, float y_percent, float width_percent, float height_percent) = 0;
      virtual View &GetView(int viewID) = 0;
      virtual void DeleteView(int viewID) = 0;

      // general
      virtual void SetCullingMode(e_CullingMode cullingMode) = 0;
      virtual void SetBlendingMode(e_BlendingMode blendingMode) = 0;
      virtual void SetDepthFunction(e_DepthFunction depthFunction) = 0;
      virtual void SetDepthTesting(bool OnOff) = 0;
      virtual void SetDepthMask(bool OnOff) = 0;
      virtual void SetBlendingFunction(e_BlendingFunction blendingFunction1, e_BlendingFunction blendingFunction2) = 0;
      virtual void SetTextureMode(e_TextureMode textureMode) = 0;
      virtual void SetColor(const Vector3 &color, float alpha) = 0;
      virtual void SetColorMask(bool r, bool g, bool b, bool alpha) = 0;

      virtual void ClearBuffer(const Vector3 &color, bool clearDepth, bool clearColor) = 0;

      virtual Matrix4 CreatePerspectiveMatrix(float aspectRatio, float nearCap = -1, float farCap = -1) = 0;
      virtual Matrix4 CreateOrthoMatrix(float left, float right, float bottom, float top, float nearCap = -1, float farCap = -1) = 0;

      // vertex buffers
      virtual VertexBufferID CreateVertexBuffer(float *vertices, unsigned int verticesDataSize, const std::vector<unsigned int>& indices, e_VertexBufferUsage usage) = 0;
      virtual void UpdateVertexBuffer(VertexBufferID vertexBufferID, float *vertices, unsigned int verticesDataSize) = 0;
      virtual void DeleteVertexBuffer(VertexBufferID vertexBufferID) = 0;
      virtual void RenderVertexBuffer(const std::deque<VertexBufferQueueEntry> &vertexBufferQueue, e_RenderMode renderMode = e_RenderMode_Full) = 0;

      // lights
      virtual void SetLight(const Vector3 &position, const Vector3 &color, float radius) = 0;

      // textures
      virtual int CreateTexture(e_InternalPixelFormat internalPixelFormat, e_PixelFormat pixelFormat, int width, int height, bool alpha = false, bool repeat = true, bool mipmaps = true, bool filter = true, bool multisample = false, bool compareDepth = false) = 0;
      virtual void ResizeTexture(int textureID, SDL_Surface *source, e_InternalPixelFormat internalPixelFormat, e_PixelFormat pixelFormat, bool alpha = false, bool mipmaps = true) = 0;
      virtual void UpdateTexture(int textureID, SDL_Surface *source, bool alpha = false, bool mipmaps = true) = 0;
      virtual void DeleteTexture(int textureID) = 0;
      virtual void CopyFrameBufferToTexture(int textureID, int width, int height) = 0;
      virtual void BindTexture(int textureID) = 0;
      virtual void SetTextureUnit(int textureUnit) = 0;
      virtual void SetClientTextureUnit(int textureUnit) = 0;

      // frame buffer
      virtual int CreateFrameBuffer() = 0;
      virtual void DeleteFrameBuffer(int fbID) = 0;
      virtual void BindFrameBuffer(int fbID) = 0;
      virtual void SetFrameBufferRenderBuffer(e_TargetAttachment targetAttachment, int rbID) = 0;
      virtual void SetFrameBufferTexture2D(e_TargetAttachment targetAttachment, int texID) = 0;
      virtual bool CheckFrameBufferStatus() = 0;
      virtual void SetFramebufferGammaCorrection(bool onOff) = 0;

      // render targets
      virtual void SetRenderTargets(std::vector<e_TargetAttachment> targetAttachments) = 0;

      // utility
      virtual void SetFOV(float angle) = 0;
      virtual void PushAttribute(int attr) = 0;
      virtual void PopAttribute() = 0;
      virtual void SetViewport(int x, int y, int width, int height) = 0;
      virtual void GetContextSize(int &width, int &height, int &bpp) = 0;

      // shaders
      virtual void LoadShader(const std::string &name, const std::string &filename) = 0;
      virtual void UseShader(const std::string &name) = 0;
      virtual void SetUniformInt(const std::string &shaderName, const std::string &varName, int value) = 0;
      virtual void SetUniformFloat(const std::string &shaderName, const std::string &varName, float value) = 0;
      virtual void SetUniformFloat2(const std::string &shaderName, const std::string &varName, float value1, float value2) = 0;
      virtual void SetUniformFloat3(const std::string &shaderName, const std::string &varName, float value1, float value2, float value3) = 0;
      virtual void SetUniformFloat3Array(const std::string &shaderName, const std::string &varName, int count, float *values) = 0;
      virtual void SetUniformMatrix4(const std::string &shaderName, const std::string &varName, const Matrix4 &mat) = 0;

    protected:
      std::map<std::string, Shader> shaders;
      std::map<std::string, Shader>::iterator currentShader;
      std::vector<View> views;

  };

class Command : public RefCounted {

    public:
      Command();
      virtual ~Command();
      bool Handle(void *caller = NULL);

    protected:
      virtual bool Execute() = 0;
  };

template <typename T = intrusive_ptr<Command> >
class MessageQueue {

    public:
      MessageQueue() { DO_VALIDATION;}
      virtual ~MessageQueue() { DO_VALIDATION;}

      inline void PushMessage(T message, bool notify = true) { DO_VALIDATION;
        queue.push_back(message);
        if (notify) NotifyWaiting();
      }

      inline void NotifyWaiting() { DO_VALIDATION;
        messageNotification.notify_one();
      }

      inline T GetMessage(bool &MsgAvail) { DO_VALIDATION;
        T message;
        if (queue.size() > 0) { DO_VALIDATION;
          message = *queue.begin();
          queue.pop_front();
          MsgAvail = true;
        } else {
          MsgAvail = false;
        }
        return message;
      }

    protected:
      std::list < T > queue;
      boost::condition messageNotification;

  };

class GraphicsSystem {

    public:
      GraphicsSystem();
      virtual ~GraphicsSystem();

      virtual void Initialize(bool render, int width_, int height_);
      virtual void Exit();
      void SetContext();
      void DisableContext();
      const screenshoot& GetScreen();

      e_SystemType GetSystemType() const;

      GraphicsScene *Create2DScene(boost::shared_ptr<IScene> scene);
      GraphicsScene *Create3DScene(boost::shared_ptr<IScene> scene);

      GraphicsTask *GetTask();
      virtual Renderer3D *GetRenderer3D();

      MessageQueue<Overlay2DQueueEntry> &GetOverlay2DQueue();

      Vector3 GetContextSize() { DO_VALIDATION; return Vector3(width, height, bpp); }

      virtual std::string GetName() const { return "graphics"; }

    protected:
      const e_SystemType systemType = e_SystemType_Graphics;

      Renderer3D *renderer3DTask = 0;

      GraphicsTask *task = 0;

      MessageQueue<Overlay2DQueueEntry> overlay2DQueue;

      int width = 0, height = 0, bpp = 0;

  };

class Scene : public IScene {

    public:
      Scene();
      virtual ~Scene();

      virtual void Init() = 0; // ATOMIC
      virtual void Exit(); // ATOMIC

      virtual void CreateSystemObjects(intrusive_ptr<Object> object);
      virtual bool SupportedObjectType(e_ObjectType objectType) const;

    protected:
      std::vector<e_ObjectType> supportedObjectTypes;
  };

typedef std::vector < intrusive_ptr<Object> > vector_Objects;
class Scene2D : public Scene {

    public:
      Scene2D(int width, int height);
      virtual ~Scene2D();

      virtual void Init();
      virtual void Exit(); // ATOMIC

      void AddObject(intrusive_ptr<Object> object);
      void DeleteObject(intrusive_ptr<Object> object);

      void PokeObjects(e_ObjectType targetObjectType,
                       e_SystemType targetSystemType);

      void GetContextSize(int &width, int &height, int &bpp);

     protected:
      vector_Objects objects;

      const int width, height, bpp;

  };

class Image2D : public Object {

    public:
      Image2D(std::string name);
      virtual ~Image2D();

      virtual void Exit();

      void SetImage(intrusive_ptr < Resource<Surface> > image);
      intrusive_ptr < Resource<Surface> > GetImage();

      void SetPosition(int x, int y);
      virtual void SetPosition(const Vector3 &newPosition, bool updateSpatialData = true);
      virtual Vector3 GetPosition() const;
      Vector3 GetSize() const;
      void DrawRectangle(int x, int y, int w, int h, const Vector3 &color,
                         int alpha = 255);
      void Resize(int w, int h);

      virtual void Poke(e_SystemType targetSystemType);

      void OnChange();

    protected:
      int position[2];
      int size[2];
      intrusive_ptr < Resource<Surface> > image;

  };

class Gui2View {

    public:
      Gui2View(Gui2WindowManager *windowManager, const std::string &name, float x_percent, float y_percent, float width_percent, float height_percent);
      virtual ~Gui2View();

      virtual void Exit();

      virtual void UpdateImagePosition();
      virtual void UpdateImageVisibility();
      virtual void AddView(Gui2View *view);
      virtual void RemoveView(Gui2View *view);
      virtual void SetParent(Gui2View *view);
      virtual Gui2View *GetParent();
      virtual void SetPosition(float x_percent, float y_percent);
      virtual void GetSize(float &width_percent, float &height_percent) const;
      virtual void GetPosition(float &x_percent, float &y_percent) const;
      virtual void GetDerivedPosition(float &x_percent, float &y_percent) const;
      virtual void CenterPosition();
      virtual void GetImages(std::vector < intrusive_ptr<Image2D> > &target);

      virtual void Redraw() { DO_VALIDATION;}

      bool IsFocussed();
      void SetFocus();
      virtual void OnGainFocus() { DO_VALIDATION; if (!children.empty()) children.at(0)->SetFocus(); }
      virtual void OnLoseFocus() { DO_VALIDATION;}
      virtual void SetInFocusPath(bool onOff) { DO_VALIDATION;
        isInFocusPath = onOff;
        if (parent) parent->SetInFocusPath(onOff);
      }

      virtual bool IsVisible() { DO_VALIDATION; if (isVisible) { DO_VALIDATION; if (parent) return parent->IsVisible(); else return true; } else return false; }
      virtual bool IsSelectable() { DO_VALIDATION; return isSelectable; }
      virtual bool IsOverlay() { DO_VALIDATION; return isOverlay; }

      virtual void Show();
      virtual void ShowAllChildren();
      virtual void Hide();

      void SetRecursiveZPriority(int prio);
      virtual void SetZPriority(int prio);

    protected:
      Gui2WindowManager *windowManager;
      std::string name;
      Gui2View *parent;

      std::vector<Gui2View*> children;
      bool exit_called = false;

      float x_percent = 0.0f;
      float y_percent = 0.0f;
      float width_percent = 0.0f;
      float height_percent = 0.0f;

      bool isVisible = false;
      bool isSelectable = false;
      bool isInFocusPath = false;
      bool isOverlay = false;
  };

class Gui2Root : public Gui2View {

    public:
      Gui2Root(Gui2WindowManager *windowManager, const std::string &name, float x_percent, float y_percent, float width_percent, float height_percent);
      virtual ~Gui2Root();

    protected:

  };

class Gui2Style {

    public:
      Gui2Style();
      virtual ~Gui2Style();

      void SetFont(e_TextType textType, TTF_Font *font);
      void SetColor(e_DecorationType decorationType, const Vector3 &color);

      TTF_Font *GetFont(e_TextType textType) const;
      Vector3 GetColor(e_DecorationType decorationType) const;

     protected:
      std::map <e_TextType, TTF_Font*> fonts;
      std::map <e_DecorationType, Vector3> colors;

  };

class Gui2Frame : public Gui2View {

    public:
      Gui2Frame(Gui2WindowManager *windowManager, const std::string &name, float x_percent, float y_percent, float width_percent, float height_percent, bool background = false);
      virtual ~Gui2Frame();

    protected:

  };

struct Gui2PageData {
    int pageID = 0;
  };

class Gui2Page : public Gui2Frame {

    public:
      Gui2Page(Gui2WindowManager *windowManager, const Gui2PageData &pageData);
      virtual ~Gui2Page();
    protected:
      Gui2PageData pageData;

  };

class Gui2PageFactory {

    public:
      Gui2PageFactory();
      virtual ~Gui2PageFactory();

      virtual void SetWindowManager(Gui2WindowManager *wm);

      virtual Gui2Page *CreatePage(int pageID, void *data = 0);
      virtual Gui2Page *CreatePage(const Gui2PageData &pageData) = 0;

    protected:
      Gui2WindowManager *windowManager;
  };

class Gui2PagePath {

    public:
      Gui2PagePath() { DO_VALIDATION;};
      virtual ~Gui2PagePath() { DO_VALIDATION; Clear(); };

      bool Empty() { DO_VALIDATION; return path.empty(); }
      void Push(const Gui2PageData &pageData,
                Gui2Page *mostRecentlyCreatedPage) { DO_VALIDATION;
        CHECK(!this->mostRecentlyCreatedPage);
        path.push_back(pageData);
        this->mostRecentlyCreatedPage = mostRecentlyCreatedPage;
      }
      void Pop() { DO_VALIDATION;
        path.pop_back();
      }
      Gui2PageData GetLast() { DO_VALIDATION;
        return path.back();
      }
      void Clear() { DO_VALIDATION;
        DeleteCurrent();
        path.clear();
      }
      Gui2Page* GetMostRecentlyCreatedPage() { DO_VALIDATION;
        return mostRecentlyCreatedPage;
      }
      void DeleteCurrent() { DO_VALIDATION;
        if (mostRecentlyCreatedPage) { DO_VALIDATION;
          mostRecentlyCreatedPage->Exit();
          delete mostRecentlyCreatedPage;
          mostRecentlyCreatedPage = nullptr;
        }
      }
    protected:
      std::vector<Gui2PageData> path;
      Gui2Page *mostRecentlyCreatedPage = nullptr;
  };

class Gui2WindowManager {

    public:
      Gui2WindowManager(boost::shared_ptr<Scene2D> scene2D, float aspectRatio, float margin);
      virtual ~Gui2WindowManager();

      void Exit();

      Gui2Root *GetRoot() { DO_VALIDATION; return root; }
      void SetFocus(Gui2View *view);

      void GetCoordinates(float x_percent, float y_percent, float width_percent,
                          float height_percent, int &x, int &y, int &width,
                          int &height) const;
      float GetWidthPercent(int pixels);
      float GetHeightPercent(int pixels);
      intrusive_ptr<Image2D> CreateImage2D(const std::string &name, int width, int height, bool sceneRegister = false);
      void UpdateImagePosition(Gui2View *view) const;
      void RemoveImage(intrusive_ptr<Image2D> image) const;

      void SetTimeStep_ms(unsigned long timeStep_ms) { DO_VALIDATION;
        this->timeStep_ms = timeStep_ms;
      };

      bool IsFocussed(Gui2View *view) { DO_VALIDATION; if (focus == view) return true; else return false; }

      Gui2Style *GetStyle() { DO_VALIDATION; return style; }

      void Show(Gui2View *view);
      void Hide(Gui2View *view);

      float GetAspectRatio() const { return aspectRatio; }

      void SetPageFactory(Gui2PageFactory *factory) { DO_VALIDATION; pageFactory = factory; factory->SetWindowManager(this); }
      Gui2PageFactory *GetPageFactory() { DO_VALIDATION; return pageFactory; }
      Gui2PagePath *GetPagePath() { DO_VALIDATION; return pagePath; }

    protected:
      boost::shared_ptr<Scene2D> scene2D;
      float aspectRatio = 0.0f;
      float margin = 0.0f;
      float effectiveW = 0.0f;
      float effectiveH = 0.0f;

      intrusive_ptr<Image2D> blackoutBackground;

      Gui2Root *root;
      Gui2View *focus;
      unsigned long timeStep_ms = 0;

      Gui2Style *style;

      Gui2PageFactory *pageFactory;
      Gui2PagePath *pagePath;

  };

class Gui2Task {

    public:
      Gui2Task(boost::shared_ptr<Scene2D> scene2D, float aspectRatio, float margin);
      ~Gui2Task();
      Gui2WindowManager *GetWindowManager() { DO_VALIDATION; return this->windowManager; }
    protected:
      Gui2WindowManager *windowManager;
  };

class Gui2Image : public Gui2View {

    public:
      Gui2Image(Gui2WindowManager *windowManager, const std::string &name, float x_percent, float y_percent, float width_percent, float height_percent);
      virtual ~Gui2Image();

      virtual void GetImages(std::vector < intrusive_ptr<Image2D> > &target);

      void LoadImage(const std::string &filename);
      virtual void Redraw();

    protected:
      intrusive_ptr<Image2D> image;
      intrusive_ptr<Image2D> imageSource;

  };

struct SideSelection {
  int controllerID = 0;
  Gui2Image *controllerImage;
  int side = 0; // -1, 0, 1
};

struct TeamTactics {
  Properties userProperties;
};


std::string real_to_str(real r) {
  DO_VALIDATION;
  std::string r_str;
  char r_c[32];
  snprintf(r_c, 32, "%f", r);
  r_str.assign(r_c);
  return r_str;
}
class PlayerProperties {
 public:
  PlayerProperties() { DO_VALIDATION;
    for (int x = 0; x < player_stat_max; x++) { DO_VALIDATION;
      data[x] = 1.0f;
    }
  }
  void Set(PlayerStat name, real value) { DO_VALIDATION;
    data[name] = atof(real_to_str(value).c_str());
  }
  real GetReal(PlayerStat name) const {
    return data[name];
  }

 private:
  real data[player_stat_max];
};

class PlayerData {

  public:
    PlayerData(int playerDatabaseID, bool left_team);
    PlayerData();
    virtual ~PlayerData();
    std::string GetLastName() const { return lastName; }
    inline float GetStat(PlayerStat name) const { return stats.GetReal(name); }
    float get_physical_velocity() const { return physical_velocity; }

    int GetSkinColor() const { return skinColor; }

    std::string GetHairStyle() const { return hairStyle; }
    void SetHairStyle(const std::string& style) { DO_VALIDATION; hairStyle = style; }

    std::string GetHairColor() const { return hairColor; }
    float GetHeight() const { return height; }

  private:
    void UpdateValues();
    float physical_velocity = 0.0;
  protected:
    PlayerProperties stats;

    int skinColor = 0;
    std::string hairStyle;
    std::string hairColor;
    float height = 0.0f;
    std::string firstName;
    std::string lastName;
};

class TeamData {

  public:
    TeamData(int teamDatabaseID, const std::vector<FormationEntry>& f);
    ~TeamData();

    std::string GetName() const { return name; }
    std::string GetShortName() const { return shortName; }
    std::string GetLogoUrl() const { return logo_url; }
    std::string GetKitUrl() const { return kit_url; }
    Vector3 GetColor1() const { return color1; }
    Vector3 GetColor2() const { return color2; }

    const TeamTactics &GetTactics() const { return tactics; }

    FormationEntry GetFormationEntry(int num) const;
    void SetFormationEntry(int num, FormationEntry entry);

    // vector index# is entry in formation[index#]
    int GetPlayerNum() const { return playerData.size(); }
    PlayerData *GetPlayerData(int num) const { return playerData.at(num); }

   protected:
    std::string name;
    std::string shortName;
    std::string logo_url;
    std::string kit_url;
    Vector3 color1, color2;

    TeamTactics tactics;

    std::vector<FormationEntry> formation;
    std::vector<PlayerData*> playerData;

};

class MatchData {

  public:
    MatchData();
    TeamData& GetTeamData(int id) { DO_VALIDATION; return teamData[id]; }
    int GetGoalCount(int id) { DO_VALIDATION; return goalCount[id]; }
    void SetGoalCount(int id, int amount) { DO_VALIDATION; goalCount[id] = amount; }
    void AddPossessionTime(int teamID, unsigned long time);
    float GetPossessionFactor_60seconds() { DO_VALIDATION;
      return possession60seconds / 60.0f;
    }  // REMEMBER THESE ARE IRL INGAME SECONDS (because, I guess the tactics
       // should be based on irl possession time instead of gametime? not sure
       // yet, think about this)
    void ProcessState(EnvState* state, int first_team);
   protected:
    TeamData teamData[2];

    int goalCount[2];

    float possession60seconds; // -60 to 60 for possession of team 1 / 2 respectively

};

struct QueuedFixture {
  QueuedFixture() { DO_VALIDATION;
    team1KitNum = 1;
    team2KitNum = 2;
    matchData = 0;
  }
  std::vector<SideSelection> sides; // queued match fixture
  int team1KitNum, team2KitNum;
  MatchData *matchData;
};

class MenuTask : public Gui2Task {

  public:
    MenuTask(float aspectRatio, float margin, TTF_Font *defaultFont, TTF_Font *defaultOutlineFont, const Properties* config);
    virtual ~MenuTask();

    void SetControllerSetup(const std::vector<SideSelection> &sides) { DO_VALIDATION; queuedFixture.sides = sides;  }
    const std::vector<SideSelection> GetControllerSetup() { DO_VALIDATION;
      return queuedFixture.sides;
    }
    int GetTeamKitNum(int teamID) { DO_VALIDATION; if (teamID == 0) return queuedFixture.team1KitNum; else return queuedFixture.team2KitNum; }
    void SetMatchData(MatchData *matchData) { DO_VALIDATION;  queuedFixture.matchData = matchData;  }
    MatchData *GetMatchData() { DO_VALIDATION; return queuedFixture.matchData; } // hint: this lock is useless, since we are returning the pointer and not a copy

  protected:
   QueuedFixture queuedFixture;

};

class Scene3D : public Scene {

    public:
      Scene3D();
      virtual ~Scene3D();

      virtual void Init();
      virtual void Exit(); // ATOMIC

      void AddNode(intrusive_ptr<Node> node);
      void DeleteNode(intrusive_ptr<Node> node);

      void GetObjects(std::deque < intrusive_ptr<Object> > &gatherObjects, const vector_Planes &bounding) const {
        hierarchyRoot->GetObjects(gatherObjects, bounding, true, 0);
      }

      template <class T>
      void GetObjects(e_ObjectType targetObjectType, std::list < intrusive_ptr<T> > &gatherObjects) const {
        if (!SupportedObjectType(targetObjectType)) { DO_VALIDATION;
          Log(e_Error, "Scene3D", "GetObjects", "targetObjectType " + int_to_str(targetObjectType) + " is not supported by this scene");
          return;
        }

        hierarchyRoot->GetObjects<T>(targetObjectType, gatherObjects, true, 0);
      }

      template <class T>
      void GetObjects(e_ObjectType targetObjectType, std::deque < intrusive_ptr<T> > &gatherObjects, const vector_Planes &bounding) const {
        if (!SupportedObjectType(targetObjectType)) { DO_VALIDATION;
          Log(e_Error, "Scene3D", "GetObjects", "targetObjectType " + int_to_str(targetObjectType) + " is not supported by this scene");
          return;
        }

        hierarchyRoot->GetObjects<T>(targetObjectType, gatherObjects, bounding, true, 0);
      }

      void PokeObjects(e_ObjectType targetObjectType, e_SystemType targetSystemType);

    protected:
      intrusive_ptr<Node> hierarchyRoot;

  };

class ObjectFactory {

    public:
      ObjectFactory();
      virtual ~ObjectFactory();

      intrusive_ptr<Object> CopyObject(intrusive_ptr<Object> source, std::string postfix = "_copy");
  };

template < typename T >
class Loader {

    public:
      virtual void Load(std::string filename, intrusive_ptr < Resource<T> > resource) = 0;

    protected:

  };

template <typename T>
class ResourceManager {

    public:
      ResourceManager() { DO_VALIDATION;};

      ~ResourceManager() { DO_VALIDATION;

        resources.clear();

        loaders.clear();
      };

      void RegisterLoader(const std::string &extension, Loader<T> *loader) { DO_VALIDATION;
        //printf("registering loader for extension %s\n", extension.c_str());
        loaders.insert(std::make_pair(extension, loader));
      }

      intrusive_ptr < Resource<T> > Fetch(const std::string &filename, bool load = true, bool useExisting = true) { DO_VALIDATION;
        bool foo = false;
        return Fetch(filename, load, foo, useExisting);
      }

       intrusive_ptr< Resource<T> > Fetch(const std::string &filename, bool load, bool &alreadyThere, bool useExisting) { DO_VALIDATION;
        std::string adaptedFilename = get_file_name(filename);

        // resource already loaded?

        bool success = false;
        intrusive_ptr < Resource<T> > foundResource;

        if (useExisting) { DO_VALIDATION;
          foundResource = Find(adaptedFilename, success);
        }

        if (success) { DO_VALIDATION;
          // resource is already there! w00t, that'll win us some cycles
          // (or user wants a new copy)

          alreadyThere = true;

          return foundResource;
        }

        else {

          // create resource

          alreadyThere = false;
          intrusive_ptr < Resource <T> > resource(new Resource<T>(adaptedFilename));

          // try to load

          if (load) { DO_VALIDATION;
            std::string extension = get_file_extension(filename);
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            typename std::map < std::string, Loader<T>* >::iterator iter = loaders.find(extension);
            if (iter != loaders.end()) { DO_VALIDATION;
              (*iter).second->Load(filename, resource);
            } else {
              Log(e_FatalError, "ResourceManager<>", "Load", "There is no loader for " + filename);
            }
          }
          Register(resource);
          return resource;
        }
      }

      intrusive_ptr < Resource<T> > FetchCopy(const std::string &filename, const std::string &newName, bool &alreadyThere) { DO_VALIDATION;
        intrusive_ptr < Resource<T> > resourceCopy;
        if (resources.find(newName) != resources.end()) { DO_VALIDATION;
          //Log(e_Warning, "ResourceManager", "FetchCopy", "Duplicate key '" + newName + "' - returning existing resource instead of copy (maybe just use Fetch() instead?)");
          resourceCopy = Fetch(newName, false, true);
        } else {
          intrusive_ptr < Resource<T> > resource = Fetch(filename, true, alreadyThere, true);

          resourceCopy = intrusive_ptr < Resource<T> >(new Resource<T>(*resource, newName));

          Register(resourceCopy);
        }

        return resourceCopy;
      }

      void RemoveUnused() { DO_VALIDATION;
        // periodically execute this cleanup code somewhere
        // currently invoked from scheduler, could be a user task?
        // as if it were a service..
        // would be slower, but somewhat cooler :p

        // cleanup



        typename std::map < std::string, intrusive_ptr< Resource<T> > >::iterator resIter = resources.begin();
        while (resIter != resources.end()) { DO_VALIDATION;
          if (resIter->second->GetRefCount() == 1) { DO_VALIDATION;
            //printf("removing unused %s resource '%s'\n", typeDescription.c_str(), resIter->second->GetIdentString().c_str());
            resources.erase(resIter++);
          } else {
            ++resIter;
          }
        }


      }

      void Remove(const std::string &identString) { DO_VALIDATION;

        auto iter = resources.find(identString);
        if (iter != resources.end()) { DO_VALIDATION;
          resources.erase(iter);
        }

      }

    protected:

      intrusive_ptr < Resource<T> > Find(const std::string &identString, bool &success) { DO_VALIDATION;

        typename std::map < std::string, intrusive_ptr< Resource<T> > >::iterator resIter = resources.find(identString);
        if (resIter != resources.end()) { DO_VALIDATION;
          success = true;
          intrusive_ptr < Resource<T> > resource = (*resIter).second;

          return resource;
        } else {
          success = false;

          return intrusive_ptr < Resource<T> >();
        }
      }

      void Register(intrusive_ptr < Resource<T> > resource) { DO_VALIDATION;



        //printf("registering %s\n", resource->GetIdentString().c_str());
        if (resources.find(resource->GetIdentString()) != resources.end()) { DO_VALIDATION;
           Remove(resource->GetIdentString());
          if (resources.find(resource->GetIdentString()) != resources.end()) { DO_VALIDATION;
            Log(e_FatalError, "ResourceManager", "Register", "Duplicate key '" + resource->GetIdentString() + "'");
          }
        }
        resources.insert(std::make_pair(resource->GetIdentString(), resource));
      }

      std::map < std::string, Loader<T>* > loaders;

      std::map < std::string, intrusive_ptr < Resource <T> > > resources;

    private:

  };

struct s_treeentry {
    std::string name;
    std::vector <std::string> values;

    s_tree *subtree;

    s_treeentry() { DO_VALIDATION;
      subtree = NULL;
    }

    ~s_treeentry();
  };

struct s_tree {
    std::vector <s_treeentry*> entries;

    ~s_tree() { DO_VALIDATION;
      for (int i = 0; i < (signed int)entries.size(); i++) { DO_VALIDATION;
        delete entries[i];
      }
      entries.clear();
    }
  };

struct s_Material {
    std::string maps[4];
    std::string shininess;
    std::string specular_amount;
    Vector3 self_illumination;
  };

class ASELoader : public Loader<GeometryData> {

    public:
      ASELoader();
      virtual ~ASELoader();

      // ----- encapsulating load function
      virtual void Load(std::string filename, intrusive_ptr < Resource <GeometryData> > resource);

      // ----- interpreter for the .ase treedata
      void Build(const s_tree *data, intrusive_ptr < Resource <GeometryData> > resource);

      // ----- per-object interpreters
      void BuildTriangleMesh(const s_tree *data, intrusive_ptr < Resource <GeometryData> > resource, std::vector <s_Material> materialList);

    protected:

      int triangleCount = 0;

  };

class ImageLoader : public Loader<Surface> {

    public:
      ImageLoader();
      virtual ~ImageLoader();

      virtual void Load(std::string filename, intrusive_ptr < Resource <Surface> > resource);

    protected:

  };

struct BiasedOffset {
    Quaternion orientation;
    float bias = 0.0f; // 0 .. 1
    void ProcessState(EnvState* state) { DO_VALIDATION;
      state->process(orientation);
      state->process(bias);
    }
  };

struct BiasedOffsets {
   public:
    BiasedOffsets() { DO_VALIDATION;
    }
    BiasedOffsets(const BiasedOffsets &obj) { DO_VALIDATION;
      for (int x = 0; x < body_part_max; x++) { DO_VALIDATION;
        elements[x] = obj.elements[x];
      }
    }
    void clear() { DO_VALIDATION;
    }
    inline BiasedOffset& operator[](BodyPart part) { DO_VALIDATION;
      return elements[part];
    }
    void ProcessState(EnvState* state) { DO_VALIDATION;
      for (auto& el : elements) { DO_VALIDATION;
        el.ProcessState(state);
      }
    }
  private:
    BiasedOffset elements[body_part_max];
  };

struct KeyFrame {
    Quaternion orientation;
    Vector3 position;
    bool operator<(const KeyFrame& a) const {
      return position < a.position;
    }
  };

struct KeyFrames {
    std::vector<std::pair<int, KeyFrame>> d;
    void clear() { DO_VALIDATION;
      d.clear();
    }
    KeyFrame* getFrame(int frame) { DO_VALIDATION;
      for (auto& i : d) { DO_VALIDATION;
        if (i.first == frame) { DO_VALIDATION;
          return &i.second;
        }
      }
      return nullptr;
    }
    void addFrame(const std::pair<int, KeyFrame>& frame) { DO_VALIDATION;
      d.push_back(frame);
      std::sort(d.begin(), d.end());
    }
  };

class AnimationExtension {

    public:
      AnimationExtension(Animation *parent) : parent(parent) { DO_VALIDATION;};
      virtual ~AnimationExtension() { DO_VALIDATION; parent = 0; };

      virtual void Shift(int fromFrame, int offset) = 0;
      virtual void Rotate2D(radian angle) = 0;
      virtual void Mirror() = 0;

      virtual bool GetKeyFrame(int frame, Quaternion &orientation, Vector3 &position, float &power) const = 0;
      virtual void SetKeyFrame(int frame, const Quaternion &orientation, const Vector3 &position = Vector3(0, 0, 0), float power = 1.0) = 0;
      virtual void DeleteKeyFrame(int frame) = 0;

      virtual void Load(std::vector<std::string> &tokenizedLine) = 0;
      virtual void Save(FILE *file) = 0;

    protected:
      Animation *parent;

  };

class VariableCache {
   public:
    std::string get(const std::string& key) const {
      auto iter = values.find(key);
      if (iter != values.end()) { DO_VALIDATION;
        return iter->second;
      } else {
        return "";
      }
    }
    void set(const std::string& key, const std::string& value) { DO_VALIDATION;
      values[key] = value;
      if (key == "idlelevel") { DO_VALIDATION;
        _idlelevel = atof(value.c_str());
      } else if (key == "quadrant_id") { DO_VALIDATION;
        _quadrant_id = atoi(value.c_str());
      } else if (key == "specialvar1") { DO_VALIDATION;
        _specialvar1 = atof(value.c_str());
      } else if (key == "specialvar2") { DO_VALIDATION;
        _specialvar2 = atof(value.c_str());
      } else if (key == "lastditch") { DO_VALIDATION;
        _lastditch = value.compare("true") == 0;
      } else if (key == "baseanim") { DO_VALIDATION;
        _baseanim = value.compare("true") == 0;
      } else if (key == "outgoing_special_state") { DO_VALIDATION;
        _outgoing_special_state = value;
      } else if (key == "incoming_retain_state") { DO_VALIDATION;
        _incoming_retain_state = value;
      } else if (key == "incoming_special_state") { DO_VALIDATION;
        _incoming_special_state = value;
      }
    }

    void set_specialvar1(float v) { DO_VALIDATION;
      _specialvar1 = v;
    }

    void set_specialvar2(float v) { DO_VALIDATION;
      _specialvar2 = v;
    }

    void mirror() { DO_VALIDATION;
      for (auto varIter : values) { DO_VALIDATION;
        mirror(varIter.second);
      }
      mirror(_outgoing_special_state);
      mirror(_incoming_retain_state);
      mirror(_incoming_special_state);
    }

    inline float idlelevel() const { return _idlelevel; }
    inline int quadrant_id() const { return _quadrant_id; }
    inline float specialvar1() const { return _specialvar1; }
    inline float specialvar2() const { return _specialvar2; }
    inline bool lastditch() const { return _lastditch; }
    inline bool baseanim() const { return _baseanim; }
    inline const std::string& outgoing_special_state() const { return _outgoing_special_state; }
    inline const std::string& incoming_retain_state() const { return _incoming_retain_state; }
    inline const std::string& incoming_special_state() const { return _incoming_special_state; }

   private:
    void mirror(std::string& varData) { DO_VALIDATION;
      if (varData.substr(0, 4) == "left") { DO_VALIDATION;
        varData = varData.replace(0, 4, "right");
      } else if (varData.substr(0, 5) == "right") { DO_VALIDATION;
        varData = varData.replace(0, 5, "left");
      }
    }

    float _idlelevel = 0;
    int _quadrant_id = 0;
    float _specialvar1 = 0;
    float _specialvar2 = 0;
    bool _lastditch = false;
    bool _baseanim = false;
    std::string _outgoing_special_state;
    std::string _incoming_retain_state;
    std::string _incoming_special_state;
    std::map<std::string, std::string> values;
  };

struct MovementHistoryEntry {
    Vector3 position;
    Quaternion orientation;
    int timeDiff_ms = 0;
    BodyPart nodeName;
    void ProcessState(EnvState* state) { DO_VALIDATION;
      state->process(position);
      state->process(orientation);
      state->process(timeDiff_ms);
      state->process(nodeName);
    }
  };

struct NodeAnimation {
    BodyPart nodeName;
    KeyFrames animation; // frame, angles
  };

typedef std::multimap<std::string, XMLTree> map_XMLTree;

struct XMLTree {
    std::string value;
    map_XMLTree children;
  };

static BiasedOffsets emptyOffsets;

class Animation {

    public:
      Animation();
      Animation(const Animation &src); // attention! this does not deep copy extensions!
      virtual ~Animation();

      void DirtyCache(); // hee hee

      int GetFrameCount() const;
      int GetEffectiveFrameCount() const { return GetFrameCount() - 1; }

      bool GetKeyFrame(BodyPart nodeName, int frame, Quaternion &orientation, Vector3 &position) const;
      void SetKeyFrame(BodyPart nodeName, int frame,
                       const Quaternion &orientation,
                       const Vector3 &position = Vector3(0, 0, 0));
      void GetInterpolatedValues(const KeyFrames &animation,
                                 int frame, Quaternion &orientation,
                                 Vector3 &position) const;
      void ConvertToStartFacingForwardIfIdle();
      void Apply(const NodeMap& nodeMap,
                 int frame, int timeOffset_ms = 0, bool smooth = true,
                 float smoothFactor = 1.0f,
                 /*const boost::shared_ptr<Animation> previousAnimation, int
                    smoothFrames, */
                 const Vector3 &basePos = Vector3(0), radian baseRot = 0,
                 BiasedOffsets &offsets = emptyOffsets,
                 MovementHistory *movementHistory = 0, int timeDiff_ms = 10,
                 bool noPos = false, bool updateSpatial = true);

      // returns end position - start position
      Vector3 GetTranslation() const;
      Vector3 GetIncomingMovement() const;
      float GetIncomingVelocity() const;
      Vector3 GetOutgoingMovement() const;
      Vector3 GetOutgoingDirection() const;
      Vector3 GetIncomingBodyDirection() const;
      Vector3 GetOutgoingBodyDirection() const;
      float GetOutgoingVelocity() const;
      radian GetOutgoingAngle() const;
      radian GetIncomingBodyAngle() const;
      radian GetOutgoingBodyAngle() const;
      e_Foot GetCurrentFoot() const { return currentFoot; }
      e_Foot GetOutgoingFoot() const;

      void Reset();
      void LoadData(std::vector < std::vector<std::string> > &file);
      void Load(const std::string &filename);
      void Mirror();
      std::string GetName() const;
      void SetName(const std::string &name) { DO_VALIDATION; this->name = name; }

      void AddExtension(const std::string &name, boost::shared_ptr<AnimationExtension> extension);
      boost::shared_ptr<AnimationExtension> GetExtension(const std::string &name);

      const std::string GetVariable(const char *name) const;
      const VariableCache& GetVariableCache() const {
        return variableCache;
      }
      void SetVariable(const std::string &name, const std::string &value);
      e_DefString GetAnimType() const { return cache_AnimType; }

      std::vector<NodeAnimation *> &GetNodeAnimations() { DO_VALIDATION;
        return nodeAnimations;
      }
      mutable float order_float = 0;
      void ProcessState(EnvState* state);

    protected:
      std::vector<NodeAnimation*> nodeAnimations;
      int frameCount = 0;
      std::string name;

      std::map < std::string, boost::shared_ptr<AnimationExtension> > extensions;

      boost::shared_ptr<XMLTree> customData;
      VariableCache variableCache;

      // this hack only applies to humanoids
      // it's which foot is moving first in this anim
      e_Foot currentFoot;

      mutable bool cache_translation_dirty = false;
      mutable Vector3 cache_translation;
      mutable bool cache_incomingMovement_dirty = false;
      mutable Vector3 cache_incomingMovement;
      mutable bool cache_incomingVelocity_dirty = false;
      mutable float cache_incomingVelocity = 0.0f;
      mutable bool cache_outgoingDirection_dirty = false;
      mutable Vector3 cache_outgoingDirection;
      mutable bool cache_outgoingMovement_dirty = false;
      mutable Vector3 cache_outgoingMovement;
      mutable bool cache_rangedOutgoingMovement_dirty = false;
      mutable Vector3 cache_rangedOutgoingMovement;
      mutable bool cache_outgoingVelocity_dirty = false;
      mutable float cache_outgoingVelocity = 0.0f;
      mutable bool cache_angle_dirty = false;
      mutable radian cache_angle;
      mutable bool cache_incomingBodyAngle_dirty = false;
      mutable radian cache_incomingBodyAngle;
      mutable bool cache_outgoingBodyAngle_dirty = false;
      mutable radian cache_outgoingBodyAngle;
      mutable bool cache_incomingBodyDirection_dirty = false;
      mutable Vector3 cache_incomingBodyDirection;
      mutable bool cache_outgoingBodyDirection_dirty = false;
      mutable Vector3 cache_outgoingBodyDirection;
      e_DefString cache_AnimType;
  };

typedef std::vector<int> DataSet;

struct CrudeSelectionQuery {
  CrudeSelectionQuery() { DO_VALIDATION;
    byFunctionType = false;
    byFoot = false; foot = e_Foot_Left;
    heedForcedFoot = false; strongFoot = e_Foot_Right;
    bySide = false;
    allowLastDitchAnims = false;
    byIncomingVelocity = false; incomingVelocity_Strict = false; incomingVelocity_NoDribbleToIdle = false; incomingVelocity_NoDribbleToSprint = false; incomingVelocity_ForceLinearity = false;
    byOutgoingVelocity = false;
    byPickupBall = false; pickupBall = true;
    byIncomingBodyDirection = false; incomingBodyDirection_Strict = false; incomingBodyDirection_ForceLinearity = false;
    byIncomingBallDirection = false;
    byOutgoingBallDirection = false;
    byTripType = false;
  }

  bool byFunctionType = false;
  e_FunctionType functionType;

  bool byFoot = false;
  e_Foot foot;

  bool heedForcedFoot = false;
  e_Foot strongFoot;

  bool bySide = false;
  Vector3 lookAtVecRel;

  bool allowLastDitchAnims = false;

  bool byIncomingVelocity = false;
  bool incomingVelocity_Strict = false; // if true, allow no difference in velocity
  bool incomingVelocity_NoDribbleToIdle = false;
  bool incomingVelocity_NoDribbleToSprint = false;
  bool incomingVelocity_ForceLinearity = false;
  e_Velocity incomingVelocity;

  bool byOutgoingVelocity = false;
  e_Velocity outgoingVelocity;

  bool byPickupBall = false;
  bool pickupBall = false;

  bool byIncomingBodyDirection = false;
  Vector3 incomingBodyDirection;
  bool incomingBodyDirection_Strict = false;
  bool incomingBodyDirection_ForceLinearity = false;

  bool byIncomingBallDirection = false;
  Vector3 incomingBallDirection;

  bool byOutgoingBallDirection = false;
  Vector3 outgoingBallDirection;

  bool byTripType = false;
  int tripType = 0;

  VariableCache properties;
};

struct Quadrant {
  int id = 0;
  Vector3 position;
  e_Velocity velocity;
  radian angle;
  void ProcessState(EnvState* state) {
    state->process(id);
    state->process(position);
    state->process(velocity);
    state->process(angle);
  }
};

class AnimCollection {

  public:
    // scene3D for debugging pilon
    AnimCollection();
    virtual ~AnimCollection();

    void Clear();
    void Load();

    const std::vector < Animation* > &GetAnimations() const;

    void CrudeSelection(DataSet &dataSet, const CrudeSelectionQuery &query);

    inline Animation* GetAnim(int index) { DO_VALIDATION;
      return animations.at(index);
    }

    inline const Quadrant &GetQuadrant(int id) { DO_VALIDATION;
      return quadrants.at(id);
    }

    int GetQuadrantID(Animation *animation, const Vector3 &movement, radian angle) const;

    void ProcessState(EnvState* state);

  protected:

    void _PrepareAnim(Animation *animation, intrusive_ptr<Node> playerNode, const std::list < intrusive_ptr<Object> > &bodyParts, const NodeMap &nodeMap, bool convertAngledDribbleToWalk = false);

    bool _CheckFunctionType(e_DefString functionType, e_FunctionType queryFunctionType) const;

    std::vector<Animation*> animations;
    std::vector<Quadrant> quadrants;

    radian maxIncomingBallDirectionDeviation;
    radian maxOutgoingBallDirectionDeviation;

};

struct BallSpatialInfo {
  BallSpatialInfo(const Vector3 &momentum, const Quaternion &rotation_ms) { DO_VALIDATION;
    this->momentum = momentum;
    this->rotation_ms = rotation_ms;
  }
  Vector3 momentum;
  Quaternion rotation_ms;
};

class Ball {

  public:
    Ball(Match *match);
    virtual ~Ball();

    void Mirror();
    intrusive_ptr<Geometry> GetBallGeom() { DO_VALIDATION; return ball; }

    inline Vector3 Predict(int predictTime_ms) const {
      int index = predictTime_ms;
      if (index >= ballPredictionSize_ms) index = ballPredictionSize_ms - 10;
      index = index / 10;
      if (index < 0) index = 0;
      return predictions[index];
    }

    void GetPredictionArray(std::vector<Vector3> &target);
    Vector3 GetMovement();
    Vector3 GetRotation();
    void Touch(const Vector3 &target);
    void SetPosition(const Vector3 &target);
    void SetMomentum(const Vector3 &target);
    void SetRotation(real x, real y, real z, float bias = 1.0);     // radians per second for each axis
    BallSpatialInfo CalculatePrediction();  // returns momentum in 10ms

    bool BallTouchesNet() { DO_VALIDATION; return ballTouchesNet; }
    Vector3 GetAveragePosition(unsigned int duration_ms) const;

    void Process();
    void Put();

    void ResetSituation(const Vector3 &focusPos);
    void ProcessState(EnvState *state);
  private:
    intrusive_ptr<Node> ballNode;
    intrusive_ptr<Geometry> ball;
    Vector3 momentum;
    Quaternion rotation_ms;

    Vector3 predictions[ballPredictionSize_ms / 10 + cachedPredictions + 1];
    int valid_predictions = 0;
    Quaternion orientPrediction;

    std::list<Vector3> ballPosHistory;

    Vector3 positionBuffer;
    Quaternion orientationBuffer;

    Match *match;

    bool ballTouchesNet = false;

};

class AIControlledKeyboard {

  public:
    AIControlledKeyboard(e_PlayerColor color);
    bool GetButton(e_ButtonFunction buttonFunction);
    void ResetNotSticky();
    void SetButton(e_ButtonFunction buttonFunction, bool state);
    bool GetPreviousButtonState(e_ButtonFunction buttonFunction);
    Vector3 GetDirection();
    Vector3 GetOriginalDirection();

    // Methods for remote controlling.
    void SetDirection(const Vector3& new_direction);
    bool Disabled() { return disabled_;}
    void SetDisabled(bool disabled);
    void Reset();
    void ProcessState(EnvState* state);
    void Mirror(float mirror);
    e_PlayerColor GetPlayerColor() const { return playerColor; }

  private:
    Vector3 direction_;
    float mirror = 1.0f;
    bool disabled_ = false;
    bool buttons_pressed_[e_ButtonFunction_Size];
    const e_PlayerColor playerColor;
};

struct TouchInfo {
  Vector3         inputDirection;
  float           inputPower = 0;

  float           autoDirectionBias = 0;
  float           autoPowerBias = 0;

  Vector3         desiredDirection; // inputdirection after pass function
  float           desiredPower = 0;
  Player          *targetPlayer = 0; // null == do not use
  Player          *forcedTargetPlayer = 0; // null == do not use
  void ProcessState(EnvState* state) { DO_VALIDATION;
    state->process(inputDirection);
    state->process(inputPower);
    state->process(autoDirectionBias);
    state->process(autoPowerBias);
    state->process(desiredDirection);
    state->process(desiredPower);
    state->process(targetPlayer);
    state->process(forcedTargetPlayer);
  }
};

struct PlayerCommand {

  /* specialVar1:

    1: happy celebration
    2: inverse celebration (feeling bad)
    3: referee showing card
  */

  PlayerCommand() { DO_VALIDATION;
    desiredFunctionType = e_FunctionType_Movement;
    useDesiredMovement = false;
    desiredVelocityFloat = idleVelocity;
    strictMovement = e_StrictMovement_Dynamic;
    useDesiredLookAt = false;
    useTripType = false;
    useDesiredTripDirection = false;
    onlyDeflectAnimsThatPickupBall = false;
    tripType = 1;
    useSpecialVar1 = false;
    specialVar1 = 0;
    useSpecialVar2 = false;
    specialVar2 = 0;
    modifier = 0;
  }

  e_FunctionType desiredFunctionType;

  bool           useDesiredMovement;
  Vector3        desiredDirection;
  e_StrictMovement strictMovement;

  float          desiredVelocityFloat;

  bool           useDesiredLookAt;
  Vector3        desiredLookAt; // absolute 'look at' position on pitch

  bool           useTouchInfo = false;
  TouchInfo      touchInfo;

  bool           onlyDeflectAnimsThatPickupBall;

  bool           useTripType;
  int            tripType; // only applicable for trip anims

  bool           useDesiredTripDirection;
  Vector3        desiredTripDirection;

  bool           useSpecialVar1;
  int            specialVar1;
  bool           useSpecialVar2;
  int            specialVar2;

  int            modifier;
  void ProcessState(EnvState* state) { DO_VALIDATION;
    state->process(desiredFunctionType);
    state->process(useDesiredMovement);
    state->process(desiredDirection);
    state->process(strictMovement);
    state->process(desiredVelocityFloat);
    state->process(useDesiredLookAt);
    state->process(desiredLookAt);
    state->process(useTouchInfo);
    touchInfo.ProcessState(state);
    state->process(onlyDeflectAnimsThatPickupBall);
    state->process(useTripType);
    state->process(tripType);
    state->process(useDesiredTripDirection);
    state->process(desiredTripDirection);
    state->process(useSpecialVar1);
    state->process(specialVar1);
    state->process(useSpecialVar2);
    state->process(specialVar2);
    state->process(modifier);
  }
};

typedef std::vector<PlayerCommand> PlayerCommandQueue;
class IController {

  public:
    IController(Match *match) : match(match) { DO_VALIDATION;};
    virtual ~IController() { DO_VALIDATION;};

    virtual void RequestCommand(PlayerCommandQueue &commandQueue) = 0;
    virtual void Process() { DO_VALIDATION;};
    virtual Vector3 GetDirection() = 0;
    virtual void ProcessState(EnvState* state) = 0;
    virtual float GetFloatVelocity() = 0;
    virtual void SetPlayer(PlayerBase *player);

    // for convenience
    PlayerBase *GetPlayer() { DO_VALIDATION; return player; }
    Match *GetMatch() { DO_VALIDATION; return match; }

    virtual int GetReactionTime_ms();

    virtual void Reset() = 0;

  protected:
    PlayerBase *player = 0;
    Match *match;
};

struct PlayerImage {
  Vector3 position;
  Vector3 directionVec;
  Vector3 movement;
  Player *player;
  e_Velocity velocity = e_Velocity_Idle;
  e_PlayerRole role;
  void ProcessState(EnvState* state) { DO_VALIDATION;
    state->process(position);
    state->process(directionVec);
    state->process(movement);
    state->process(player);
    state->process(velocity);
    state->process(role);
  }
  void Mirror() { DO_VALIDATION;
    position.Mirror();
    directionVec.Mirror();
    movement.Mirror();
  }
};

struct PlayerImagePosition {
  PlayerImagePosition(const Vector3& position, const Vector3& movement, e_PlayerRole player_role) : position(position), movement(movement), player_role(player_role) { DO_VALIDATION;}
  Vector3 position;
  Vector3 movement;
  e_PlayerRole player_role;
};

class MentalImage {
 public:
  MentalImage() { DO_VALIDATION;}
  MentalImage(Match *match);
  void Mirror(bool team_0, bool team_1, bool ball);
  PlayerImage GetPlayerImage(PlayerBase* player) const;
  std::vector<PlayerImagePosition> GetTeamPlayerImages(int teamID) const;
  void UpdateBallPredictions();
  Vector3 GetBallPrediction(int time_ms) const;
  int GetTimeStampNeg_ms() const;
  void ProcessState(EnvState* state, Match* match);

  std::vector<PlayerImage> players;
  std::vector<Vector3> ballPredictions;
  unsigned int timeStamp_ms = 0;
  float maxDistanceDeviation = 2.5f;
  float maxMovementDeviation = walkVelocity;
  bool ballPredictions_mirrored = false;
 private:
  Match *match = nullptr;
};

class PlayerController : public IController {

  public:
    PlayerController(Match *match);
    virtual ~PlayerController() { DO_VALIDATION;};

    virtual void Process();

    virtual void SetPlayer(PlayerBase *player);
    Player *CastPlayer();
    Team *GetTeam() { DO_VALIDATION; return team; }
    Team *GetOppTeam() { DO_VALIDATION; return oppTeam; }

    const MentalImage *GetMentalImage();

    virtual int GetReactionTime_ms();

    float GetLastSwitchBias();

    float GetFadingTeamPossessionAmount() { DO_VALIDATION; return fadingTeamPossessionAmount; }

    void AddDefensiveComponent(Vector3 &desiredPosition, float bias, Player* forcedOpp = 0);
    Vector3 GetDefendPosition(Player *opp, float distance = 0.0f);

    virtual void Reset();
    void ProcessPlayerController(EnvState *state);

  protected:
    float OppBetweenBallAndMeDot();
    float CouldWinABallDuelLikeliness();
    virtual void _Preprocess();
    virtual void _SetInput(const Vector3 &inputDirection, float inputVelocityFloat) { DO_VALIDATION; this->inputDirection = inputDirection; this->inputVelocityFloat = inputVelocityFloat; }
    virtual void _KeeperDeflectCommand(PlayerCommandQueue &commandQueue, bool onlyPickupAnims = false);
    virtual void _SetPieceCommand(PlayerCommandQueue &commandQueue);
    virtual void _BallControlCommand(PlayerCommandQueue &commandQueue, bool idleTurnToOpponentGoal = false, bool knockOn = false, bool stickyRunDirection = false, bool keepCurrentBodyDirection = false);
    virtual void _TrapCommand(PlayerCommandQueue &commandQueue, bool idleTurnToOpponentGoal = false, bool knockOn = false);
    virtual void _InterfereCommand(PlayerCommandQueue &commandQueue, bool byAnyMeans = false);
    virtual void _SlidingCommand(PlayerCommandQueue &commandQueue);
    virtual void _MovementCommand(PlayerCommandQueue &commandQueue, bool forceMagnet = false, bool extraHaste = false);

    Vector3 inputDirection;
    float inputVelocityFloat = 0.0f;

    Player *_oppPlayer = nullptr;
    float _timeNeeded_ms = 0;
    int _mentalImageTime;

    void _CalculateSituation();

    // only really useful for human gamers, after switching player
    unsigned long lastSwitchTime_ms = 0;
    unsigned int lastSwitchTimeDuration_ms = 0;

    Team *team = nullptr;
    Team *oppTeam = nullptr;

    bool hasPossession = false;
    bool hasUniquePossession = false;
    bool teamHasPossession = false;
    bool teamHasUniquePossession = false;
    bool oppTeamHasPossession = false;
    bool oppTeamHasUniquePossession = false;
    bool hasBestPossession = false;
    bool teamHasBestPossession = false;
    float possessionAmount = 0.0f;
    float teamPossessionAmount = 0.0f;
    float fadingTeamPossessionAmount = 0.0f;
    int oppTimeNeededToGetToBall = 0;
    bool hasBestChanceOfPossession = false;
};

class HumanController : public PlayerController {

  public:
    HumanController(Match *match = nullptr, AIControlledKeyboard *hid = nullptr);
    virtual ~HumanController();

    virtual void SetPlayer(PlayerBase *player);
    bool Disabled() const {
      return hid->Disabled();
    }

    virtual void RequestCommand(PlayerCommandQueue &commandQueue);
    virtual void Process();
    virtual Vector3 GetDirection();
    virtual float GetFloatVelocity();

    void PreProcess(Match *match, AIControlledKeyboard *hid) {
      this->match = match;
      this->hid = hid;
   }

    void ProcessState(EnvState* state) { DO_VALIDATION;
      ProcessPlayerController(state);
      hid->ProcessState(state);
      state->process(actionMode);
      state->process(actionButton);
      state->process(actionBufferTime_ms);
      state->process(gauge_ms);
      state->process(previousDirection);
      state->process(steadyDirection);
      state->process(lastSteadyDirectionSnapshotTime_ms);
    }
    virtual int GetReactionTime_ms();

    AIControlledKeyboard *GetHIDevice() { return hid; }

    int GetActionMode() { DO_VALIDATION; return actionMode; }

    virtual void Reset();

  protected:

    void _GetHidInput(Vector3 &rawInputDirection, float &rawInputVelocityFloat);

    AIControlledKeyboard *hid;

    // set when a contextual button (example: pass/defend button) is pressed
    // once this is set and the button stays pressed, it stays the same
    // 0: undefined, 1: off-the-ball button active, 2: on-the-ball button active/action queued
    int actionMode = 0;

    e_ButtonFunction actionButton;
    int actionBufferTime_ms = 0;
    int gauge_ms = 0;

    // stuff to keep track of analog stick (or keys even) so that we can use a direction once it's been pointed in for a while, instead of directly
    Vector3 previousDirection;
    Vector3 steadyDirection;
    int lastSteadyDirectionSnapshotTime_ms = 0;
    float mirror = 1.0;
};

class HumanGamer {

  public:
    HumanGamer(Team *team, AIControlledKeyboard *hid);
    HumanGamer() {}
    HumanGamer(const HumanGamer&) = delete;
    void operator=(const HumanGamer&) = delete;
    ~HumanGamer();

    Player *GetSelectedPlayer() const {
      return selectedPlayer;
    }
    void SetSelectedPlayer(Player* player);
    AIControlledKeyboard *GetHIDevice() { DO_VALIDATION; return hid; }
    HumanController* GetHumanController() { DO_VALIDATION; return &controller; }
    void ProcessState(EnvState *state);

  protected:
    Team *team = nullptr;
    AIControlledKeyboard *hid = nullptr;
    HumanController controller;
    Player *selectedPlayer = nullptr;
};

struct TacticalOpponentInfo {
  Player *player;
  float dangerFactor = 0.0f;
  void ProcessState(EnvState* state) { DO_VALIDATION;
    state->process(player);
    state->process(dangerFactor);
  }
};

class TeamAIController {

  public:
    TeamAIController(Team *team);
    virtual ~TeamAIController();

    void Process();

    Vector3 GetAdaptedFormationPosition(Player *player, bool useDynamicFormationPosition = true);
    void CalculateDynamicRoles();
    float CalculateMarkingQuality(Player *player, Player *opp);
    void CalculateManMarking();
    void ApplyOffsideTrap(Vector3 &position) const;
    float GetOffsideTrapX() const { return offsideTrapX; }
    void PrepareSetPiece(e_GameMode setPiece, Team* other_team,
                         int kickoffTakerTeamId, int takerTeamID);
    Player *GetPieceTaker() { DO_VALIDATION; return taker; }
    e_GameMode GetSetPieceType() { DO_VALIDATION; return setPieceType; }
    void ApplyAttackingRun(Player *manualPlayer = 0);
    void ApplyTeamPressure();
    void ApplyKeeperRush();
    void CalculateSituation();

    void UpdateTactics();

    unsigned long GetEndApplyAttackingRun_ms() { DO_VALIDATION; return endApplyAttackingRun_ms; }
    Player *GetAttackingRunPlayer() { DO_VALIDATION; return attackingRunPlayer; }

    unsigned long GetEndApplyTeamPressure_ms() { DO_VALIDATION; return endApplyTeamPressure_ms; }
    Player *GetTeamPressurePlayer() { DO_VALIDATION; return teamPressurePlayer; }

    Player *GetForwardSupportPlayer() { DO_VALIDATION; return forwardSupportPlayer; }

    unsigned long GetEndApplyKeeperRush_ms() { DO_VALIDATION; return endApplyKeeperRush_ms; }

    const std::vector<TacticalOpponentInfo> &GetTacticalOpponentInfo() { DO_VALIDATION; return tacticalOpponentInfo; }

    void Reset();
    void ProcessState(EnvState* state);

  protected:

    Match *match;
    Team *team;
    Player *taker;
    e_GameMode setPieceType;

    Properties baseTeamTactics;
    Properties teamTacticsModMultipliers;
    Properties liveTeamTactics;

    float offensivenessBias = 0.0f;

    bool teamHasPossession = false;
    bool teamHasUniquePossession = false;
    bool oppTeamHasPossession = false;
    bool oppTeamHasUniquePossession = false;
    bool teamHasBestPossession = false;
    float teamPossessionAmount = 0.0f;
    float fadingTeamPossessionAmount = 0.0f;
    int timeNeededToGetToBall = 0;
    int oppTimeNeededToGetToBall = 0;

    float depth = 0.0f;
    float width = 0.0f;

    float offsideTrapX = 0.0f;

    unsigned long endApplyAttackingRun_ms = 0;
    Player *attackingRunPlayer;
    unsigned long endApplyTeamPressure_ms = 0;
    Player *teamPressurePlayer;
    unsigned long endApplyKeeperRush_ms = 0;

    Player *forwardSupportPlayer; // sort of like the attacking run player, but more for a forward offset for a player close to the action, to support the player in possession

    std::vector<TacticalOpponentInfo> tacticalOpponentInfo;

};

class Team {

  public:
    Team(int id, Match *match, TeamData *teamData, float aiDifficulty);
    void Mirror();
    bool isMirrored() { DO_VALIDATION;
      return mirrored;
    }
    bool onOriginalSide() { DO_VALIDATION;
      return id == 0 ? (side == -1) : (side == 1);
    }

    virtual ~Team();

    void Exit();

    void InitPlayers(intrusive_ptr<Node> fullbodyNode,
                     std::map<Vector3, Vector3> &colorCoords);

    Match *GetMatch() { DO_VALIDATION; return match; }
    TeamAIController *GetController() { DO_VALIDATION; return teamController; }
    intrusive_ptr<Node> GetSceneNode() { DO_VALIDATION; return teamNode; }

    int GetID() const { return id; }
    inline signed int GetDynamicSide() { DO_VALIDATION;
      return side;
    }
    inline signed int GetStaticSide() { DO_VALIDATION;
      return id == 0 ? -1 : 1;
    }
    const TeamData *GetTeamData() { DO_VALIDATION; return teamData; }

    FormationEntry GetFormationEntry(void* player);
    void SetFormationEntry(Player* player, FormationEntry entry);
    float GetAiDifficulty() const { return aiDifficulty; }
    const std::vector<Player *> &GetAllPlayers() { return players; }
    void GetAllPlayers(std::vector<Player*> &allPlayers) { DO_VALIDATION;
      allPlayers.insert(allPlayers.end(), players.begin(), players.end());
    }
    void GetActivePlayers(std::vector<Player *> &activePlayers);
    int GetActivePlayersCount() const;
    Player *MainSelectedPlayer() { return mainSelectedPlayer; }

    unsigned int GetHumanGamerCount() {
      int count = 0;
      for (auto& g: humanGamers) { DO_VALIDATION;
        if (!g->GetHumanController()->Disabled()) {
          count++;
        }
      }
      return count;
    }
    void GetHumanControllers(std::vector<HumanGamer*>& v) {
      for (auto& g: humanGamers) { DO_VALIDATION;
        v.push_back(g.get());
      }
    }
    void AddHumanGamers(const std::vector<AIControlledKeyboard*>& controllers);
    void DeleteHumanGamers();
    e_PlayerColor GetPlayerColor(PlayerBase* player);
    int HumanControlledToBallDistance();

    bool HasPossession() const;
    bool HasUniquePossession() const;
    int GetTimeNeededToGetToBall_ms() const;
    Player *GetDesignatedTeamPossessionPlayer() { DO_VALIDATION;
      return designatedTeamPossessionPlayer;
    }
    void UpdateDesignatedTeamPossessionPlayer();
    Player *GetBestPossessionPlayer();
    float GetTeamPossessionAmount() const;
    float GetFadingTeamPossessionAmount() const;
    void SetFadingTeamPossessionAmount(float value);

    void SetLastTouchPlayer(
        Player *player, e_TouchType touchType = e_TouchType_Intentional_Kicked);
    Player *GetLastTouchPlayer() const { return lastTouchPlayer; }
    float GetLastTouchBias(int decay_ms, unsigned long time_ms = 0) { DO_VALIDATION;
      return lastTouchPlayer
                 ? lastTouchPlayer->GetLastTouchBias(decay_ms, time_ms)
                 : 0;
    }

    void ResetSituation(const Vector3 &focusPos);

    void HumanGamersSelectAnyone();
    void SetOpponent(Team* opponent) { DO_VALIDATION; this->opponent = opponent; }
    Team* Opponent() { DO_VALIDATION; return opponent; }
    void SelectPlayer(Player *player);
    void DeselectPlayer(Player *player);

    void RelaxFatigue(float howMuch);

    void Process();
    void PreparePutBuffers();
    void FetchPutBuffers();
    void Put(bool mirror);
    void Put2D(bool mirror);
    void Hide2D();

    void UpdatePossessionStats();
    void UpdateSwitch();
    void ProcessState(EnvState* state);

    Player *GetGoalie();

  protected:
    const int id;
    Match *match;
    Team *opponent = 0;
    TeamData *teamData;
    const float aiDifficulty;

    bool hasPossession = false;
    int timeNeededToGetToBall_ms = 0;
    Player *designatedTeamPossessionPlayer = 0;

    float teamPossessionAmount = 0.0f;
    float fadingTeamPossessionAmount = 0.0f;

    TeamAIController *teamController;

    std::vector<Player*> players;

    intrusive_ptr<Node> teamNode;
    intrusive_ptr<Node> playerNode;

    std::vector<std::unique_ptr<HumanGamer>> humanGamers;

    // humanGamers index whose turn it is
    // begin() == due next
    std::list<int> switchPriority;
    Player *lastTouchPlayer = nullptr;
    Player *mainSelectedPlayer = nullptr;

    intrusive_ptr < Resource<Surface> > kit;
    int side = -1;
    bool mirrored = false;
};

struct RefereeBuffer {
  // Referee has pending action to execute.
  bool active = false;
  e_GameMode desiredSetPiece;
  signed int teamID = 0;
  Team* setpiece_team = 0;
  unsigned long stopTime = 0;
  unsigned long prepareTime = 0;
  unsigned long startTime = 0;
  Vector3 restartPos;
  Player *taker;
  bool endPhase = false;
  void ProcessState(EnvState* state);
};

struct Foul {
  Player *foulPlayer = 0;
  Player *foulVictim = 0;
  int foulType = 0; // 0: nothing, 1: foul, 2: yellow, 3: red
  bool advantage = false;
  unsigned long foulTime = 0;
  Vector3 foulPosition;
  bool hasBeenProcessed = false;
  void ProcessState(EnvState* state);
};

class Referee {

  public:
    Referee(Match *match, bool animations);
    virtual ~Referee();

    void Process();

    void PrepareSetPiece(e_GameMode setPiece);

    const RefereeBuffer &GetBuffer() { DO_VALIDATION; return buffer; };

    void AlterSetPiecePrepareTime(unsigned long newTime_ms);

    void BallTouched();
    void TripNotice(Player *tripee, Player *tripper, int tackleType); // 1 == standing tackle resulting in little trip, 2 == standing tackle resulting in fall, 3 == sliding tackle
    bool CheckFoul();

    Player *GetCurrentFoulPlayer() { DO_VALIDATION; return foul.foulPlayer; }
    int GetCurrentFoulType() { DO_VALIDATION; return foul.foulType; }
    void ProcessState(EnvState* state);

  protected:
    Match *match;

    RefereeBuffer buffer;

    int afterSetPieceRelaxTime_ms = 0; // throw-ins cause immediate new throw-ins, because ball is still outside the lines at the moment of throwing ;)

    // Players on offside position at the time of the last ball touch.
    std::vector<Player*> offsidePlayers;

    Foul foul;
    const bool animations;
};

struct RotationSmuggle {
  RotationSmuggle() { DO_VALIDATION;
    begin = 0;
    end = 0;
  }
  void operator = (const float &value) { DO_VALIDATION;
    begin = value;
    end = value;
  }
  radian begin;
  radian end;
  void ProcessState(EnvState* state) { DO_VALIDATION;
    state->process(begin);
    state->process(end);
  }
};

struct Anim {
  Animation *anim = 0;
  signed int id = 0;
  int frameNum = 0;
  e_FunctionType functionType = e_FunctionType_None;
  e_InterruptAnim originatingInterrupt = e_InterruptAnim_None;
  Vector3 actionSmuggle;
  Vector3 actionSmuggleOffset;
  Vector3 actionSmuggleSustain;
  Vector3 actionSmuggleSustainOffset;
  Vector3 movementSmuggle;
  Vector3 movementSmuggleOffset;
  RotationSmuggle rotationSmuggle;
  radian rotationSmuggleOffset = 0;
  signed int touchFrame = -1;
  Vector3 touchPos;
  Vector3 incomingMovement;
  Vector3 outgoingMovement;
  Vector3 positionOffset;
  PlayerCommand originatingCommand;
  std::vector<Vector3> positions;
  void ProcessState(EnvState* state) { DO_VALIDATION;
    state->process(anim);
    state->process(id);
    state->process(frameNum);
    state->process(functionType);
    state->process(originatingInterrupt);
    state->process(actionSmuggle);
    state->process(actionSmuggleOffset);
    state->process(actionSmuggleSustain);
    state->process(actionSmuggleSustainOffset);
    state->process(movementSmuggle);
    state->process(movementSmuggleOffset);
    rotationSmuggle.ProcessState(state);
    state->process(rotationSmuggleOffset);
    state->process(touchFrame);
    state->process(touchPos);
    state->process(incomingMovement);
    state->process(outgoingMovement);
    state->process(positionOffset);
    originatingCommand.ProcessState(state);
    state->process(positions);
  }
};

struct FloatArray {
  float *data;
  int size = 0;
};

struct WeightedBone {
  int jointID = 0;
  float weight = 0.0f;
};

struct WeightedVertex {
  int vertexID = 0;
  std::vector<WeightedBone> bones;
};

struct HJoint {
  intrusive_ptr<Node> node;
  Vector3 position;
  Quaternion orientation;
  Vector3 origPos;
};

struct RotationSmuggle {
  RotationSmuggle() { DO_VALIDATION;
    begin = 0;
    end = 0;
  }
  void operator = (const float &value) { DO_VALIDATION;
    begin = value;
    end = value;
  }
  radian begin;
  radian end;
  void ProcessState(EnvState* state) { DO_VALIDATION;
    state->process(begin);
    state->process(end);
  }
};

struct AnimApplyBuffer {
  AnimApplyBuffer() { DO_VALIDATION;
    frameNum = 0;
    smooth = true;
    smoothFactor = 0.5f;
    noPos = false;
    orientation = 0;
  }
  AnimApplyBuffer(const AnimApplyBuffer &src) { DO_VALIDATION;
    anim = src.anim;
    frameNum = src.frameNum;
    smooth = src.smooth;
    smoothFactor = src.smoothFactor;
    noPos = src.noPos;
    position = src.position;
    orientation = src.orientation;
    offsets = src.offsets;
  }
  void ProcessState(EnvState* state) { DO_VALIDATION;
    state->process(anim);
    state->process(frameNum);
    state->process(smooth);
    state->process(smoothFactor);
    state->process(noPos);
    state->process(position);
    state->process(orientation);
    offsets.ProcessState(state);
  }
  Animation *anim = 0;
  int frameNum = 0;
  bool smooth = false;
  float smoothFactor = 0.0f;
  bool noPos = false;
  Vector3 position;
  radian orientation;
  BiasedOffsets offsets;
};

struct SpatialState {
  Vector3 position;
  radian angle;
  Vector3 directionVec; // for efficiency, vector version of angle
  e_Velocity enumVelocity;
  float floatVelocity = 0.0f; // for efficiency, float version

  Vector3 actualMovement;
  Vector3 physicsMovement; // ignores effects like positionoffset
  Vector3 animMovement;
  Vector3 movement; // one of the above (default)
  Vector3 actionSmuggleMovement;
  Vector3 movementSmuggleMovement;
  Vector3 positionOffsetMovement;

  radian bodyAngle;
  Vector3 bodyDirectionVec; // for efficiency, vector version of bodyAngle
  radian relBodyAngleNonquantized;
  radian relBodyAngle;
  Vector3 relBodyDirectionVec; // for efficiency, vector version of relBodyAngle
  Vector3 relBodyDirectionVecNonquantized;
  e_Foot foot;

  void Mirror() { DO_VALIDATION;
    position.Mirror();
    actualMovement.Mirror();
    physicsMovement.Mirror();
    animMovement.Mirror();
    movement.Mirror();
    actionSmuggleMovement.Mirror();
    movementSmuggleMovement.Mirror();
    positionOffsetMovement.Mirror();
  }

  void ProcessState(EnvState* state) { DO_VALIDATION;
    state->process(position);
    state->process(angle);
    state->process(directionVec);
    state->process(enumVelocity);
    state->process(floatVelocity);
    state->process(actualMovement);
    state->process(physicsMovement);
    state->process(animMovement);
    state->process(movement);
    state->process(actionSmuggleMovement);
    state->process(movementSmuggleMovement);
    state->process(positionOffsetMovement);
    state->process(bodyAngle);
    state->process(bodyDirectionVec);
    state->process(relBodyAngleNonquantized);
    state->process(relBodyAngle);
    state->process(relBodyDirectionVec);
    state->process(relBodyDirectionVecNonquantized);
    state->process(foot);
  }
};

class HumanoidBase {

  public:
    HumanoidBase(PlayerBase *player, Match *match, intrusive_ptr<Node> humanoidSourceNode, intrusive_ptr<Node> fullbodySourceNode, std::map<Vector3, Vector3> &colorCoords, boost::shared_ptr<AnimCollection> animCollection, intrusive_ptr<Node> fullbodyTargetNode, intrusive_ptr < Resource<Surface> > kit);
    virtual ~HumanoidBase();
    void Mirror();

    void PrepareFullbodyModel(std::map<Vector3, Vector3> &colorCoords);
    void UpdateFullbodyNodes(bool mirror);
    void UpdateFullbodyModel(bool updateSrc = false);

    virtual void Process();
    void PreparePutBuffers();
    void FetchPutBuffers();
    void Put(bool mirror);

    virtual void CalculateGeomOffsets();
    void SetOffset(BodyPart body_part, float bias, const Quaternion &orientation, bool isRelative = false);

    inline int GetFrameNum() { DO_VALIDATION; return currentAnim.frameNum; }
    inline int GetFrameCount() { DO_VALIDATION; return currentAnim.anim->GetFrameCount(); }

    inline Vector3 GetPosition() const { return spatialState.position; }
    inline Vector3 GetDirectionVec() const { return spatialState.directionVec; }
    inline Vector3 GetBodyDirectionVec() const {
      return spatialState.bodyDirectionVec;
    }
    inline radian GetRelBodyAngle() const { return spatialState.relBodyAngle; }
    inline e_Velocity GetEnumVelocity() const { return spatialState.enumVelocity; }
    inline e_FunctionType GetCurrentFunctionType() const { return currentAnim.functionType; }
    inline e_FunctionType GetPreviousFunctionType() const { return previousAnim_functionType; }
    inline Vector3 GetMovement() const { return spatialState.movement; }

    Vector3 GetGeomPosition() { DO_VALIDATION; return humanoidNode->GetPosition(); }

    int GetIdleMovementAnimID();
    void ResetPosition(const Vector3 &newPos, const Vector3 &focusPos);
    void OffsetPosition(const Vector3 &offset);
    void TripMe(const Vector3 &tripVector, int tripType);

    intrusive_ptr<Node> GetHumanoidNode() { DO_VALIDATION; return humanoidNode; }
    intrusive_ptr<Node> GetFullbodyNode() { DO_VALIDATION; return fullbodyNode; }

    virtual float GetDecayingPositionOffsetLength() const { return decayingPositionOffset.GetLength(); }
    virtual float GetDecayingDifficultyFactor() const { return decayingDifficultyFactor; }

    const Anim *GetCurrentAnim() { DO_VALIDATION; return &currentAnim; }

    const NodeMap &GetNodeMap() { DO_VALIDATION; return nodeMap; }

    void Hide() { DO_VALIDATION; fullbodyNode->SetPosition(Vector3(1000, 1000, -1000)); hairStyle->SetPosition(Vector3(1000, 1000, -1000)); } // hax ;)

    void SetKit(intrusive_ptr < Resource<Surface> > newKit);

    virtual void ResetSituation(const Vector3 &focusPos);
    void ProcessState(EnvState* state);

  protected:
    bool _HighOrBouncyBall() const;
    void _KeepBestDirectionAnims(DataSet& dataset, const PlayerCommand &command, bool strict = true, radian allowedAngle = 0, int allowedVelocitySteps = 0, int forcedQuadrantID = -1); // ALERT: set sorting predicates before calling this function. strict kinda overrules the allowedstuff
    void _KeepBestBodyDirectionAnims(DataSet& dataset, const PlayerCommand &command, bool strict = true, radian allowedAngle = 0); // ALERT: set sorting predicates before calling this function. strict kinda overrules the allowedstuff
    virtual bool SelectAnim(const PlayerCommand &command, e_InterruptAnim localInterruptAnim, bool preferPassAndShot = false); // returns false on no applicable anim found
    void CalculatePredictedSituation(Vector3 &predictedPos, radian &predictedAngle);
    Vector3 CalculateOutgoingMovement(const std::vector<Vector3> &positions) const;

    void CalculateSpatialState(); // realtime properties, based on 'physics'
    void CalculateFactualSpatialState(); // realtime properties, based on anim. usable at last frame of anim. more riggid than above function

    void AddTripCommandToQueue(PlayerCommandQueue &commandQueue, const Vector3 &tripVector, int tripType);
    PlayerCommand GetTripCommand(const Vector3 &tripVector, int tripType);
    PlayerCommand GetBasicMovementCommand(const Vector3 &desiredDirection, float velocityFloat);

    void SetFootSimilarityPredicate(e_Foot desiredFoot) const;
    bool CompareFootSimilarity(e_Foot foot, int animIndex1, int animIndex2) const;
    void SetIncomingVelocitySimilarityPredicate(e_Velocity velocity) const;
    bool CompareIncomingVelocitySimilarity(int animIndex1, int animIndex2) const;
    void SetMovementSimilarityPredicate(const Vector3 &relDesiredDirection, e_Velocity desiredVelocity) const;
    float GetMovementSimilarity(int animIndex, const Vector3 &relDesiredDirection, e_Velocity desiredVelocity, float corneringBias) const;
    bool CompareMovementSimilarity(int animIndex1, int animIndex2) const;
    bool CompareByOrderFloat(int animIndex1, int animIndex2) const;
    void SetIncomingBodyDirectionSimilarityPredicate(
        const Vector3 &relIncomingBodyDirection) const;
    bool CompareIncomingBodyDirectionSimilarity(int animIndex1, int animIndex2) const;
    void SetBodyDirectionSimilarityPredicate(const Vector3 &lookAt) const;
    real DirectionSimilarityRating(int animIndex) const;
    bool CompareBodyDirectionSimilarity(int animIndex1, int animIndex2) const;
    void SetTripDirectionSimilarityPredicate(const Vector3 &relDesiredTripDirection) const;
    bool CompareTripDirectionSimilarity(int animIndex1, int animIndex2) const;
    bool CompareBaseanimSimilarity(int animIndex1, int animIndex2) const;
    bool CompareCatchOrDeflect(int animIndex1, int animIndex2) const;
    void SetIdlePredicate(float desiredValue) const;
    bool CompareIdleVariable(int animIndex1, int animIndex2) const;
    bool ComparePriorityVariable(int animIndex1, int animIndex2) const;

    Vector3 CalculatePhysicsVector(Animation *anim, bool useDesiredMovement,
                                   const Vector3 &desiredMovement,
                                   bool useDesiredBodyDirection,
                                   const Vector3 &desiredBodyDirectionRel,
                                   std::vector<Vector3> &positions_ret,
                                   radian &rotationOffset_ret) const;

    Vector3 ForceIntoAllowedBodyDirectionVec(const Vector3 &src) const;
    radian ForceIntoAllowedBodyDirectionAngle(radian angle) const; // for making small differences irrelevant while sorting
    Vector3 ForceIntoPreferredDirectionVec(const Vector3 &src) const;
    radian ForceIntoPreferredDirectionAngle(radian angle) const;

    // Seems to be used for rendering only, updated in
    // UpdateFullbodyModel / UpdateFullBodyNodes, Hide() method changes
    // position, so maybe Hide needs to change, otherwise collision detection
    // analysis hidden players?
    intrusive_ptr<Node> fullbodyNode;
    // Modified in PrepareFullBodyModel, not changed later.
    std::vector<FloatArray> uniqueFullbodyMesh;
    // Modified in PrepareFullBodyModel, not changed later.
    std::vector < std::vector<WeightedVertex> > weightedVerticesVec;
    // Modified in PrepareFullBodyModel, not changed later.
    unsigned int fullbodySubgeomCount = 0;
    // Used only for memory releasing.
    std::vector<int*> uniqueIndicesVec;
    // Updated in UpdateFullbodyModel / UpdateFullBodyNodes,
    // snapshot not needed. References nodes point to humanoidNode.
    std::vector<HJoint> joints;
    // Used only for memory management.
    intrusive_ptr<Node> fullbodyTargetNode;
    // Used for ball collision detection. Seems to be the one to snapshot.
    intrusive_ptr<Node> humanoidNode;
    // Updated in UpdateFullbodyNodes, no need to snapshot.
    intrusive_ptr<Geometry> hairStyle;
    // Initiated in the constructor, no need to snapshot.
    std::string kitDiffuseTextureIdentString = "kit_template.png";

    Match *match;
    PlayerBase *player;
    // Shared between all players, no need to snapshot.
    boost::shared_ptr<AnimCollection> anims;
    // Pointers from elements in humanoidNode to Nodes.
    NodeMap nodeMap;
    // Seems to contain current animation context.
    AnimApplyBuffer animApplyBuffer;

    BiasedOffsets offsets;

    Anim currentAnim;
    int previousAnim_frameNum;
    e_FunctionType previousAnim_functionType = e_FunctionType_None;

    // position/rotation offsets at the start of currentAnim
    Vector3 startPos;
    radian startAngle;

    // position/rotation offsets at the end of currentAnim
    Vector3 nextStartPos;
    radian nextStartAngle;

    // realtime info
    SpatialState spatialState;

    Vector3 previousPosition2D;

    e_InterruptAnim interruptAnim;
    int reQueueDelayFrames = 0;
    int tripType = 0;
    Vector3 tripDirection;

    Vector3 decayingPositionOffset;
    float decayingDifficultyFactor = 0.0f;

    // for comparing dataset entries (needed by std::list::sort)
    mutable e_Foot predicate_DesiredFoot;
    mutable e_Velocity predicate_IncomingVelocity;
    mutable Vector3 predicate_RelDesiredDirection;
    mutable Vector3 predicate_DesiredDirection;
    mutable float predicate_CorneringBias = 0.0f;
    mutable e_Velocity predicate_DesiredVelocity;
    mutable Vector3 predicate_RelIncomingBodyDirection;
    mutable Vector3 predicate_LookAt;
    mutable Vector3 predicate_RelDesiredTripDirection;
    mutable Vector3 predicate_RelDesiredBallDirection;
    mutable float predicate_idle = 0.0f;

    // Should be dynamically retrieved from match, don't cache.
    int mentalImageTime = 0;

    const float zMultiplier;
    MovementHistory movementHistory;
    bool mirrored = false;
};

struct ForceSpot {
  Vector3 origin;
  e_MagnetType magnetType;
  e_DecayType decayType;
  float exp = 1.0f;
  float power = 0.0f;
  float scale = 0.0f; // scaled #meters until effect is almost decimated
};

class RefereeController : public IController {

  public:
    RefereeController(Match *match);
    virtual ~RefereeController();

    PlayerOfficial *CastPlayer();

    void GetForceField(std::vector<ForceSpot> &forceField);

    virtual void RequestCommand(PlayerCommandQueue &commandQueue);
    virtual void Process();
    virtual void ProcessState(EnvState* state) { DO_VALIDATION;
    }
    virtual Vector3 GetDirection();
    virtual float GetFloatVelocity();

    virtual int GetReactionTime_ms();

    virtual void Reset();
};

class PlayerOfficial : public PlayerBase {

  public:
    PlayerOfficial(e_OfficialType officialType, Match *match, PlayerData *playerData);
    virtual ~PlayerOfficial();

    HumanoidBase *CastHumanoid();
    RefereeController *CastController();

    e_OfficialType GetOfficialType() { DO_VALIDATION; return officialType; }

    virtual void Activate(intrusive_ptr<Node> humanoidSourceNode, intrusive_ptr<Node> fullbodySourceNode, std::map<Vector3, Vector3> &colorCoords, intrusive_ptr < Resource<Surface> > kit, boost::shared_ptr<AnimCollection> animCollection, bool lazyPlayer);
    virtual void Deactivate();

    virtual void Process();
    virtual void FetchPutBuffers();

  protected:
    e_OfficialType officialType;

};

class Officials {

  public:
    Officials(Match *match, intrusive_ptr<Node> fullbodySourceNode, std::map<Vector3, Vector3> &colorCoords, intrusive_ptr < Resource<Surface> > kit, boost::shared_ptr<AnimCollection> animCollection);
    ~Officials();
    void Mirror();

    void GetPlayers(std::vector<PlayerBase*> &players);
    PlayerOfficial *GetReferee() { DO_VALIDATION; return referee; }

    void Process();
    void FetchPutBuffers();
    void Put(bool mirror);

    intrusive_ptr<Geometry> GetYellowCardGeom() { DO_VALIDATION; return yellowCard; }
    intrusive_ptr<Geometry> GetRedCardGeom() { DO_VALIDATION; return redCard; }
    void ProcessState(EnvState* state);

  protected:
    Match *match;

    PlayerOfficial *referee;
    PlayerOfficial *linesmen[2];
    PlayerData *playerData;

    intrusive_ptr<Geometry> yellowCard;
    intrusive_ptr<Geometry> redCard;

};

struct SharedInfo {
  Position ball_position;
  Position ball_direction;
  Position ball_rotation;
  std::vector<PlayerInfo> left_team;
  std::vector<PlayerInfo> right_team;
  std::vector<ControllerInfo> left_controllers;
  std::vector<ControllerInfo> right_controllers;
  int left_goals, right_goals;
  e_GameMode game_mode;
  bool is_in_play = false;
  int ball_owned_team = 0;
  int ball_owned_player = 0;
  int step = 0;
};

class Gui2Caption : public Gui2View {

    public:
      Gui2Caption(Gui2WindowManager *windowManager, const std::string &name, float x_percent, float y_percent, float width_percent, float height_percent, const std::string &caption);
      virtual ~Gui2Caption();

      virtual void GetImages(std::vector < intrusive_ptr<Image2D> > &target);

      void SetColor(const Vector3 &color);
      void SetOutlineColor(const Vector3 &outlineColor);
      void SetTransparency(float trans);

      virtual void Redraw();

      void SetCaption(const std::string &newCaption);

      float GetTextWidthPercent() { DO_VALIDATION; return textWidth_percent; }

     protected:
      intrusive_ptr<Image2D> image;

      std::string caption;
      Vector3 color;
      Vector3 outlineColor;
      float transparency = 0.0f;
      float textWidth_percent = 0.0f;
      int renderedTextHeightPix = 0;

  };

class Gui2ScoreBoard : public Gui2View {

  public:
    Gui2ScoreBoard(Gui2WindowManager *windowManager, Match *match);
    virtual ~Gui2ScoreBoard();

    void GetImages(std::vector < intrusive_ptr<Image2D> > &target);

    virtual void Redraw();

    void SetTimeStr(const std::string &timeStr);
    void SetGoalCount(int teamID, int goalCount);

  protected:
    intrusive_ptr<Image2D> image;
    Gui2Caption *timeCaption;
    Gui2Caption *teamNameCaption[2];
    Gui2Caption *goalCountCaption[2];
    Gui2Image *leagueLogo;
    Gui2Image *teamLogo[2];
};

class Gui2Radar : public Gui2View {

    public:
      Gui2Radar(Gui2WindowManager *windowManager, const std::string &name, float x_percent, float y_percent, float width_percent, float height_percent, Match *match, const Vector3 &color1_1, const Vector3 &color1_2, const Vector3 &color2_1, const Vector3 &color2_2);
      virtual ~Gui2Radar();

      void ReloadAvatars(int teamID, unsigned int playerCount);

      virtual void Process();
      void Put();

    protected:
      Gui2Image *bg;
      std::vector<Gui2Image*> team1avatars;
      std::vector<Gui2Image*> team2avatars;
      Gui2Image* ball;

      Match *match;

      Vector3 color1_1, color1_2, color2_1, color2_2;

  };

template <typename T> class ValueHistory {

    public:
      ValueHistory(unsigned int maxTime_ms = 10000) : maxTime_ms(maxTime_ms) { DO_VALIDATION;}
      virtual ~ValueHistory() { DO_VALIDATION;}

      void Insert(const T &value) { DO_VALIDATION;
        values.push_back(value);
        if (values.size() > maxTime_ms / 10) values.pop_front();
      }

      T GetAverage(unsigned int time_ms) const {
        T total = 0;
        unsigned int count = 0;
        if (!values.empty()) { DO_VALIDATION;
          typename std::list<T>::const_iterator iter = values.end();
          iter--;
          while (count <= time_ms / 10) { DO_VALIDATION;
            total += (*iter);
            count++;
            if (iter == values.begin()) break; else iter--;
          }
        }
        if (count > 0) total /= (float)count;
        return total;
      }

      void Clear() { DO_VALIDATION;
        values.clear();
      }
      void ProcessState(EnvState *state) { DO_VALIDATION;
        state->process(maxTime_ms);
        state->process(values);
      }

    protected:
      unsigned int maxTime_ms = 0;
      std::list<T> values;

  };

struct PlayerBounce {
  Player *opp;
  float force = 0.0f;
};

class Match {

  public:
    Match(MatchData *matchData, const std::vector<AIControlledKeyboard*> &controllers, bool init_animation);
    virtual ~Match();

    void Exit();
    void Mirror(bool team_0, bool team_1, bool ball);

    void SetRandomSunParams();
    void RandomizeAdboards(intrusive_ptr<Node> stadiumNode);
    void UpdateControllerSetup();
    void SpamMessage(const std::string &msg, int time_ms = 3000);
    int GetScore(int teamID) { DO_VALIDATION; return matchData->GetGoalCount(teamID); }
    Ball *GetBall() { DO_VALIDATION; return ball; }
    Team *GetTeam(int teamID) { DO_VALIDATION; return teams[teamID]; }
    void GetActiveTeamPlayers(int teamID, std::vector<Player*> &players);
    void GetOfficialPlayers(std::vector<PlayerBase*> &players);
    boost::shared_ptr<AnimCollection> GetAnimCollection();

    MentalImage* GetMentalImage(int history_ms);
    void UpdateLatestMentalImageBallPredictions();

    void ResetSituation(const Vector3 &focusPos);

    void SetMatchPhase(e_MatchPhase newMatchPhase);
    e_MatchPhase GetMatchPhase() const { return matchPhase; }

    void StartPlay() { DO_VALIDATION; inPlay = true; }
    void StopPlay() { DO_VALIDATION; inPlay = false; }
    bool IsInPlay() const { return inPlay; }

    void StartSetPiece() { DO_VALIDATION; inSetPiece = true; }
    void StopSetPiece() { DO_VALIDATION; inSetPiece = false; }
    bool IsInSetPiece() const { return inSetPiece; }
    Referee *GetReferee() { DO_VALIDATION; return referee; }
    Officials *GetOfficials() { DO_VALIDATION; return officials; }

    void SetGoalScored(bool onOff) { DO_VALIDATION; if (onOff == false) ballIsInGoal = false; goalScored = onOff; }
    bool IsGoalScored() const { return goalScored; }
    Team* GetLastGoalTeam() const { return lastGoalTeam; }
    void SetLastTouchTeamID(int id, e_TouchType touchType = e_TouchType_Intentional_Kicked) { DO_VALIDATION; lastTouchTeamIDs[touchType] = id; lastTouchTeamID = id; referee->BallTouched(); }
    int GetLastTouchTeamID(e_TouchType touchType) const { return lastTouchTeamIDs[touchType]; }
    int GetLastTouchTeamID() const { return lastTouchTeamID; }
    Team *GetLastTouchTeam() { DO_VALIDATION;
      if (lastTouchTeamID != -1)
        return teams[lastTouchTeamID];
      else
        return teams[first_team];
    }
    Player *GetLastTouchPlayer() { DO_VALIDATION;
      if (GetLastTouchTeam())
        return GetLastTouchTeam()->GetLastTouchPlayer();
      else
        return 0;
    }
    float GetLastTouchBias(int decay_ms, unsigned long time_ms = 0) { DO_VALIDATION; if (GetLastTouchTeam()) return GetLastTouchTeam()->GetLastTouchBias(decay_ms, time_ms); else return 0; }
    bool IsBallInGoal() const { return ballIsInGoal; }

    Team* GetBestPossessionTeam();

    Player *GetDesignatedPossessionPlayer() { DO_VALIDATION; return designatedPossessionPlayer; }
    Player *GetBallRetainer() { DO_VALIDATION; return ballRetainer; }
    void SetBallRetainer(Player *retainer) { DO_VALIDATION;
      ballRetainer = retainer;
    }

    float GetAveragePossessionSide(int time_ms) const { return possessionSideHistory.GetAverage(time_ms); }

    unsigned long GetMatchTime_ms() const { return matchTime_ms; }
    unsigned long GetActualTime_ms() const { return actualTime_ms; }
    void BumpActualTime_ms(unsigned long time);
    void UpdateIngameCamera();


    intrusive_ptr<Camera> GetCamera() { DO_VALIDATION; return camera; }
    void GetTeamState(SharedInfo *state, std::map<AIControlledKeyboard*, int>& controller_mapping, int team_id);
    void GetState(SharedInfo* state);
    void ProcessState(EnvState* state);
    bool Process();
    void UpdateCamera();
    void PreparePutBuffers();
    void FetchPutBuffers();
    void Put();

    intrusive_ptr<Node> GetDynamicNode();

    void FollowCamera(Quaternion &orientation, Quaternion &nodeOrientation, Vector3 &position, float &FOV, const Vector3 &targetPosition, float zoom);

    void SetAutoUpdateIngameCamera(bool autoUpdate = true) { DO_VALIDATION; if (autoUpdate != autoUpdateIngameCamera) { DO_VALIDATION; camPos.clear(); autoUpdateIngameCamera = autoUpdate; } }

    int GetReplaySize_ms();

    MatchData* GetMatchData() { DO_VALIDATION; return matchData; }

    float GetMatchDurationFactor() const { return matchDurationFactor; }
    bool GetUseMagnet() const { return _useMagnet; }

    const std::vector<Vector3> &GetAnimPositionCache(Animation *anim) const;

    void UploadGoalNetting();

    int FirstTeam() { DO_VALIDATION; return first_team; }
    int SecondTeam() { DO_VALIDATION; return second_team; }
    bool isBallMirrored() { DO_VALIDATION; return ball_mirrored; }

  private:
    bool CheckForGoal(signed int side, const Vector3& previousBallPos);

    void CalculateBestPossessionTeamID();
    void CheckHumanoidCollisions();
    void CheckHumanoidCollision(Player *p1, Player *p2, std::vector<PlayerBounce> &p1Bounce, std::vector<PlayerBounce> &p2Bounce);
    void CheckBallCollisions();

    void PrepareGoalNetting();
    void UpdateGoalNetting(bool ballTouchesNet = false);

    MatchData *matchData;
    Team *teams[2];
    int first_team = 0;
    int second_team = 1;
    bool ball_mirrored = false;

    Officials *officials;

    intrusive_ptr<Node> dynamicNode;

    intrusive_ptr<Node> cameraNode;
    intrusive_ptr<Camera> camera;
    intrusive_ptr<Node> sunNode;

    intrusive_ptr<Node> stadiumNode;

    const std::vector<AIControlledKeyboard*> &controllers;

    Ball *ball = nullptr;

    std::vector<MentalImage> mentalImages; // [index] == index * 10 ms ago ([0] == now)

    Gui2ScoreBoard *scoreboard;
    Gui2Radar *radar;
    Gui2Caption *messageCaption;
    unsigned long messageCaptionRemoveTime_ms = 0;
    unsigned long matchTime_ms = 0;
    unsigned long actualTime_ms = 0;
    unsigned long goalScoredTimer = 0;

    e_MatchPhase matchPhase = e_MatchPhase_PreMatch; // 0 - first half; 1 - second half; 2 - 1st extra time; 3 - 2nd extra time; 4 - penalties
    bool inPlay = false;
    bool inSetPiece = false; // Whether game is in special mode (corner etc...)
    bool goalScored = false; // true after goal scored, false again after next match state change
    bool ballIsInGoal = false;
    Team* lastGoalTeam = 0;
    Player *lastGoalScorer;
    int lastTouchTeamIDs[e_TouchType_SIZE];
    int lastTouchTeamID = 0;
    Team* bestPossessionTeam = 0;
    Player *designatedPossessionPlayer;
    Player *ballRetainer;

    ValueHistory<float> possessionSideHistory;

    bool autoUpdateIngameCamera = false;

    // camera
    Quaternion cameraOrientation;
    Quaternion cameraNodeOrientation;
    Vector3 cameraNodePosition;
    float cameraFOV = 0.0f;
    float cameraNearCap = 0.0f;
    float cameraFarCap = 0.0f;

    unsigned int lastBodyBallCollisionTime_ms = 0;

    std::deque<Vector3> camPos;

    Referee *referee;

    boost::shared_ptr<MenuTask> menuTask;

    boost::shared_ptr<Scene3D> scene3D;

    bool resetNetting = false;
    bool nettingHasChanged = false;

    const float matchDurationFactor = 0.0f;

    std::vector<Vector3> nettingMeshesSrc[2];
    std::vector<float*> nettingMeshes[2];
    // Whether to use magnet logic (that automatically pushes active player
    // towards the ball).
    const bool _useMagnet;
};

class PlayerBase {

  public:
    PlayerBase(Match *match, PlayerData *playerData);
    virtual ~PlayerBase();
    void Mirror();

    inline int GetStableID() const { return stable_id; }
    inline const PlayerData* GetPlayerData() { DO_VALIDATION; return playerData; }

    inline bool IsActive() { DO_VALIDATION; return isActive; }

    // get ready for some action
    virtual void Activate(intrusive_ptr<Node> humanoidSourceNode, intrusive_ptr<Node> fullbodySourceNode, std::map<Vector3, Vector3> &colorCoords, intrusive_ptr < Resource<Surface> > kit, boost::shared_ptr<AnimCollection> animCollection, bool lazyPlayer) = 0;
    // go back to bench/take a shower
    virtual void Deactivate();

    void ResetPosition(const Vector3 &newPos, const Vector3 &focusPos) { DO_VALIDATION; humanoid->ResetPosition(newPos, focusPos); }
    void OffsetPosition(const Vector3 &offset) { DO_VALIDATION; humanoid->OffsetPosition(offset); }

    inline int GetFrameNum() { DO_VALIDATION; return humanoid->GetFrameNum(); }
    inline int GetFrameCount() { DO_VALIDATION; return humanoid->GetFrameCount(); }

    inline Vector3 GetPosition() const { return humanoid->GetPosition(); }
    inline Vector3 GetGeomPosition() const { return humanoid->GetGeomPosition(); }
    inline Vector3 GetDirectionVec() const { return humanoid->GetDirectionVec(); }
    inline Vector3 GetBodyDirectionVec() const { return humanoid->GetBodyDirectionVec(); }
    inline Vector3 GetMovement() const { return humanoid->GetMovement(); }
    inline radian GetRelBodyAngle() const {
      return humanoid->GetRelBodyAngle();
    }
    inline e_Velocity GetEnumVelocity() const { return humanoid->GetEnumVelocity(); }
    inline float GetFloatVelocity() const { return EnumToFloatVelocity(humanoid->GetEnumVelocity()); }
    inline e_FunctionType GetCurrentFunctionType() const { return humanoid->GetCurrentFunctionType(); }
    inline e_FunctionType GetPreviousFunctionType() const { return humanoid->GetPreviousFunctionType(); }

    void TripMe(const Vector3 &tripVector, int tripType) { DO_VALIDATION; humanoid->TripMe(tripVector, tripType); }

    void RequestCommand(PlayerCommandQueue &commandQueue);
    IController *GetController();
    void SetExternalController(HumanGamer *externalController);
    HumanController *ExternalController();
    bool ExternalControllerActive();

    intrusive_ptr<Node> GetHumanoidNode() { DO_VALIDATION; return humanoid->GetHumanoidNode(); }
    intrusive_ptr<Node> GetFullbodyNode() { DO_VALIDATION; return humanoid->GetFullbodyNode(); }

    float GetDecayingPositionOffsetLength() { DO_VALIDATION; return humanoid->GetDecayingPositionOffsetLength(); }

    virtual void Process();
    virtual void PreparePutBuffers();
    virtual void FetchPutBuffers();
    void Put(bool mirror);

    void UpdateFullbodyModel() { DO_VALIDATION; humanoid->UpdateFullbodyModel(); }

    virtual float GetStat(PlayerStat name) const;
    float GetVelocityMultiplier() const;
    float GetMaxVelocity() const;

    const Anim *GetCurrentAnim() { DO_VALIDATION; return humanoid->GetCurrentAnim(); }

    void SetLastTouchTime_ms(unsigned long touchTime_ms) { DO_VALIDATION; this->lastTouchTime_ms = touchTime_ms; }
    unsigned long GetLastTouchTime_ms() { DO_VALIDATION; return lastTouchTime_ms; }
    void SetLastTouchType(e_TouchType touchType) { DO_VALIDATION; this->lastTouchType = touchType; }
    e_TouchType GetLastTouchType() { DO_VALIDATION; return lastTouchType; }
    float GetLastTouchBias(int decay_ms, unsigned long time_ms = 0);

    const NodeMap &GetNodeMap() { DO_VALIDATION; return humanoid->GetNodeMap(); }

    float GetFatigueFactorInv() const { return fatigueFactorInv; }
    void RelaxFatigue(float howMuch) { DO_VALIDATION;
      fatigueFactorInv += howMuch;
      fatigueFactorInv = clamp(fatigueFactorInv, 0.01f, 1.0f);
    }

    virtual void ResetSituation(const Vector3 &focusPos);

    void ProcessStateBase(EnvState* state);

  protected:
    Match *match;

    const PlayerData* const playerData;
    const int stable_id = 0;

    std::unique_ptr<HumanoidBase> humanoid;
    std::unique_ptr<IController> controller;
    HumanGamer *externalController = 0;

    bool isActive = false;

    unsigned long lastTouchTime_ms = 0;
    e_TouchType lastTouchType;

    float fatigueFactorInv = 0.0f;

    std::vector<Vector3> positionHistoryPerSecond; // resets too (on ResetSituation() calls)

};

class Humanoid : public HumanoidBase {

  public:
    Humanoid(Player *player, intrusive_ptr<Node> humanoidSourceNode, intrusive_ptr<Node> fullbodySourceNode, std::map<Vector3, Vector3> &colorCoords, boost::shared_ptr<AnimCollection> animCollection, intrusive_ptr<Node> fullbodyTargetNode, intrusive_ptr < Resource<Surface> > kit);
    virtual ~Humanoid();

    Player *CastPlayer() const;

    virtual void Process();

    virtual void CalculateGeomOffsets();

    bool TouchPending() { DO_VALIDATION; return (currentAnim.frameNum < currentAnim.touchFrame) ? true : false; }
    bool TouchAnim() { DO_VALIDATION; return (currentAnim.touchFrame != -1) ? true : false; }
    Vector3 GetTouchPos() { DO_VALIDATION; return currentAnim.touchPos; }
    int GetTouchFrame() { DO_VALIDATION; return currentAnim.touchFrame; }
    int GetCurrentFrame() { DO_VALIDATION; return currentAnim.frameNum; }

    void SelectRetainAnim();

    virtual void ResetSituation(const Vector3 &focusPos);

  protected:
    virtual bool SelectAnim(const PlayerCommand &command, e_InterruptAnim localInterruptAnim, bool preferPassAndShot = false); // returns false on no applicable anim found
    bool NeedTouch(int animID, const PlayerCommand &command);
    float GetBodyBallDistanceAdvantage(
        const Animation *anim, e_FunctionType functionType,
        const Vector3 &animTouchMovement, const Vector3 &touchMovement,
        const Vector3 &incomingMovement, const Vector3 &outgoingMovement,
        radian outgoingAngle,
        /*const Vector3 &animBallToBall2D, */ const Vector3 &bodyPos,
        const Vector3 &FFO, const Vector3 &animBallPos2D,
        const Vector3 &actualBallPos2D, const Vector3 &ballMovement2D,
        float radiusFactor, float radiusCheatDistance, float decayPow,
        bool debug = false) const;
    signed int GetBestCheatableAnimID(const DataSet &sortedDataSet, bool useDesiredMovement, const Vector3 &desiredDirection, float desiredVelocityFloat, bool useDesiredBodyDirection, const Vector3 &desiredBodyDirectionRel, std::vector<Vector3> &positions_ret, int &animTouchFrame_ret, float &radiusOffset_ret, Vector3 &touchPos_ret, Vector3 &fullActionSmuggle_ret, Vector3 &actionSmuggle_ret, radian &rotationSmuggle_ret, e_InterruptAnim localInterruptAnim, bool preferPassAndShot = false) const;
    Vector3 CalculateMovementSmuggle(const Vector3 &desiredDirection, float desiredVelocityFloat);
    Vector3 GetBestPossibleTouch(const Vector3 &desiredTouch, e_FunctionType functionType);

    Team *team;
};

class DefaultDefenseStrategy {
  public:
    void RequestInput(ElizaController *controller, const MentalImage *mentalImage, Vector3 &direction, float &velocity);
};

class DefaultMidfieldStrategy {

  public:
    void RequestInput(ElizaController *controller, const MentalImage *mentalImage, Vector3 &direction, float &velocity);
};

class DefaultOffenseStrategy {

  public:
    void RequestInput(ElizaController *controller, const MentalImage *mentalImage, Vector3 &direction, float &velocity);

  protected:

};

class GoalieDefaultStrategy {
  public:
    void RequestInput(ElizaController *controller, const MentalImage *mentalImage, Vector3 &direction, float &velocity);
    void CalculateIfBallIsBoundForGoal(ElizaController *controller, const MentalImage *mentalImage);
    bool IsBallBoundForGoal() const { return ballBoundForGoal; }
    void ProcessState(EnvState* state);

  protected:
    bool ballBoundForGoal = false;
    float ballBoundForGoal_ycoord = 0.0f;
};

class ElizaController : public PlayerController {

  public:
    ElizaController(Match *match, bool lazyPlayer);
    virtual ~ElizaController();

    virtual void RequestCommand(PlayerCommandQueue &commandQueue);
    virtual void Process();
    virtual Vector3 GetDirection();
    virtual float GetFloatVelocity();

    float GetLazyVelocity(float desiredVelocityFloat);
    Vector3 GetSupportPosition_ForceField(const MentalImage *mentalImage,
                                          const Vector3 &basePosition,
                                          bool makeRun = false);

    virtual void Reset();
    virtual void ProcessState(EnvState* state);

  protected:
    void GetOnTheBallCommands(std::vector<PlayerCommand> &commandQueue, Vector3 &rawInputDirection, float &rawInputVelocity);

    void _AddPass(std::vector<PlayerCommand> &commandQueue, Player *target, e_FunctionType passType);
    void _AddPanicPass(std::vector<PlayerCommand> &commandQueue);
    float _GetPassingOdds(Player *targetPlayer, e_FunctionType passType, const std::vector<PlayerImagePosition> &opponentPlayerImages, float ballVelocityMultiplier = 1.0f);
    float _GetPassingOdds(const Vector3 &target, e_FunctionType passType, const std::vector<PlayerImagePosition> &opponentPlayerImages, float ballVelocityMultiplier = 1.0f);
    void _AddCelebration(std::vector<PlayerCommand> &commandQueue);

    DefaultDefenseStrategy defenseStrategy;
    DefaultMidfieldStrategy midfieldStrategy;
    DefaultOffenseStrategy offenseStrategy;
    GoalieDefaultStrategy goalieStrategy;

    Vector3 lastDesiredDirection;
    float lastDesiredVelocity = 0.0f;
    const bool lazyPlayer = false;
};

struct TacticalPlayerSituation {
  float forwardSpaceRating = 0.0f;
  float toGoalSpaceRating = 0.0f;
  float spaceRating = 0.0f;
  float forwardRating = 0.0f;
  void ProcessState(EnvState* state) { DO_VALIDATION;
    state->process(forwardSpaceRating);
    state->process(toGoalSpaceRating);
    state->process(spaceRating);
    state->process(forwardRating);
  }
};

class Player : public PlayerBase {

  public:
    Player(Team *team, PlayerData *playerData);
    virtual ~Player();

    Humanoid *CastHumanoid();
    ElizaController *CastController();

    int GetTeamID() const;
    Team *GetTeam();
    Vector3 GetPitchPosition();

    // get ready for some action
    virtual void Activate(intrusive_ptr<Node> humanoidSourceNode, intrusive_ptr<Node> fullbodySourceNode, std::map<Vector3, Vector3> &colorCoords, intrusive_ptr < Resource<Surface> > kit, boost::shared_ptr<AnimCollection> animCollection, bool lazyPlayer);
    // go back to bench/take a shower
    virtual void Deactivate();

    bool TouchPending() { DO_VALIDATION; return CastHumanoid()->TouchPending(); }
    bool TouchAnim() { DO_VALIDATION; return CastHumanoid()->TouchAnim(); }
    Vector3 GetTouchPos() { DO_VALIDATION; return CastHumanoid()->GetTouchPos(); }
    int GetTouchFrame() { DO_VALIDATION; return CastHumanoid()->GetTouchFrame(); }
    int GetCurrentFrame() { DO_VALIDATION; return CastHumanoid()->GetCurrentFrame(); }

    void SelectRetainAnim() { DO_VALIDATION; CastHumanoid()->SelectRetainAnim(); }

    inline e_FunctionType GetCurrentFunctionType() { DO_VALIDATION; return CastHumanoid()->GetCurrentFunctionType(); }
    FormationEntry GetFormationEntry();
    inline void SetDynamicFormationEntry(FormationEntry entry) { DO_VALIDATION; dynamicFormationEntry = entry; }
    inline FormationEntry GetDynamicFormationEntry() { DO_VALIDATION; return dynamicFormationEntry; }
    inline void SetManMarking(Player* player) { DO_VALIDATION; manMarking = player; }
    inline Player* GetManMarking() { DO_VALIDATION; return manMarking; }

    bool HasPossession() const;
    bool HasBestPossession() const;
    bool HasUniquePossession() const;
    inline int GetPossessionDuration_ms() const { return possessionDuration_ms; }
    inline int GetTimeNeededToGetToBall_ms() const { return timeNeededToGetToBall_ms; }
    inline int GetTimeNeededToGetToBall_optimistic_ms() const { return timeNeededToGetToBall_optimistic_ms; }
    inline int GetTimeNeededToGetToBall_previous_ms() const { return timeNeededToGetToBall_previous_ms; }
    void SetDesiredTimeToBall_ms(int ms) { DO_VALIDATION; desiredTimeToBall_ms = ms; }
    int GetDesiredTimeToBall_ms() const { return clamp(desiredTimeToBall_ms, timeNeededToGetToBall_ms, 1000000.0f); }
    bool AllowLastDitch(bool includingPossessionAmount = true) const;

    void TriggerControlledBallCollision() { DO_VALIDATION; triggerControlledBallCollision = true; }
    bool IsControlledBallCollisionTriggered() { DO_VALIDATION; return triggerControlledBallCollision; }
    void ResetControlledBallCollisionTrigger() { DO_VALIDATION; triggerControlledBallCollision = false; }

    float GetAverageVelocity(float timePeriod_sec); // is reset on ResetSituation() calls

    void UpdatePossessionStats();

    float GetClosestOpponentDistance() const;

    const TacticalPlayerSituation &GetTacticalSituation() { DO_VALIDATION; return tacticalSituation; }

    virtual void Process();
    virtual void PreparePutBuffers();
    virtual void FetchPutBuffers();
    void Put2D(bool mirror);
    void Hide2D();

    void GiveYellowCard(unsigned long giveTime_ms) { DO_VALIDATION; cards++; cardEffectiveTime_ms = giveTime_ms; }
    void GiveRedCard(unsigned long giveTime_ms) { DO_VALIDATION;
      cards += 3;
      cardEffectiveTime_ms = giveTime_ms;
    }

    bool HasCards() { DO_VALIDATION;
      return cards > 0;
    }

    void SendOff();

    float GetStaminaStat() const;
    virtual float GetStat(PlayerStat name) const;

    virtual void ResetSituation(const Vector3 &focusPos);

    void ProcessState(EnvState* state);
  protected:
    void _CalculateTacticalSituation();

    Team *team = nullptr;

    Player* manMarking = 0;

    FormationEntry dynamicFormationEntry;

    bool hasPossession = false;
    bool hasBestPossession = false;
    bool hasUniquePossession = false;
    int possessionDuration_ms = 0;
    unsigned int timeNeededToGetToBall_ms = 1000;
    unsigned int timeNeededToGetToBall_optimistic_ms = 1000;
    unsigned int timeNeededToGetToBall_previous_ms = 1000;

    bool triggerControlledBallCollision = false;

    TacticalPlayerSituation tacticalSituation;

    bool buf_nameCaptionShowCondition = false;
    Vector3 buf_playerColor;

    Gui2Caption *nameCaption = nullptr;

    int desiredTimeToBall_ms = 0;
    int cards = 0; // 1 == 1 yellow; 2 == 2 yellow; 3 == 1 red; 4 == 1 yellow, 1 red

    unsigned long cardEffectiveTime_ms = 0;

};

class GameTask {

  public:
    GameTask();
    ~GameTask();

    void StartMatch(bool init_animation);
    bool StopMatch();

    void ProcessPhase();
    void PrepareRender();

    Match *GetMatch() { DO_VALIDATION; return match.get(); }

  protected:
    std::unique_ptr<Match> match;
    boost::shared_ptr<Scene3D> scene3D;
};

class GameContext {
 public:
  GameContext() : rng(BaseGenerator(), Distribution()), rng_non_deterministic(BaseGenerator(), Distribution()) { }
  GraphicsSystem graphicsSystem;
  boost::shared_ptr<GameTask> gameTask;
  boost::shared_ptr<MenuTask> menuTask;
  boost::shared_ptr<Scene2D> scene2D;
  boost::shared_ptr<Scene3D> scene3D;
  intrusive_ptr<Node> fullbodyNode;
  intrusive_ptr<Node> goalsNode;
  intrusive_ptr<Node> stadiumRender;
  intrusive_ptr<Node> stadiumNoRender;
  Properties *config = nullptr;
  std::string font;
  TTF_Font *defaultFont = nullptr;
  TTF_Font *defaultOutlineFont = nullptr;

  std::vector<AIControlledKeyboard*> controllers;
  ObjectFactory object_factory;
  ResourceManager<GeometryData> geometry_manager;
  ResourceManager<Surface> surface_manager;
  ResourceManager<Texture> texture_manager;
  ResourceManager<VertexBuffer> vertices_manager;
  ASELoader aseLoader;
  ImageLoader imageLoader;

  typedef boost::mt19937 BaseGenerator;
  typedef boost::uniform_real<float> Distribution;
  typedef boost::variate_generator<BaseGenerator, Distribution> Generator;
  Generator rng;

  // Two random number generators are needed. One (deterministic when running
  // in deterministic mode) to be used in places which generate deterministic
  // game state. Second one is used in places which are optional and don't
  // affect observations (like position of the sun).
  Generator rng_non_deterministic;
  bool already_loaded = false;
  int playerCount = 0;
  int stablePlayerCount = 0;
  BiasedOffsets emptyOffsets;
  boost::shared_ptr<AnimCollection> anims;
  std::map<Animation*, std::vector<Vector3>> animPositionCache;
  std::map<Vector3, Vector3> colorCoords;
  int step = 0;
  int tracker_disabled = 1;
  long tracker_pos = 0;
  void ProcessState(EnvState* state);
};

class EnvState {
 public:
  EnvState(GameEnv* game_env, const std::string& state, const std::string reference = "");
  const ScenarioConfig* getConfig() { return scenario_config; }
  const GameContext* getContext() { return context; }
  void process(std::string &value);
  void process(Animation* &value);
  template<typename T> void process(std::vector<T>& collection) {
    int size = collection.size();
    process(size);
    collection.resize(size);
    for (auto& el : collection) {
      process(el);
    }
  }
  template<typename T> void process(std::list<T>& collection) {
    int size = collection.size();
    process(size);
    collection.resize(size);
    for (auto& el : collection) {
      process(el);
    }
  }
  void process(Player*& value);
  void process(HumanGamer*& value);
  void process(AIControlledKeyboard*& value);
  void process(Team*& value);
  bool isFailure() {
    return failure;
  }
  bool enabled() {
    return this->disable_cnt == 0;
  }
  void setValidate(bool validate) {
    this->disable_cnt += validate ? -1 : 1;
  }
  void setCrash(bool crash) {
    this->crash = crash;
  }
  bool Load() { return load; }
  int getpos() {
    return pos;
  }
  bool eos();
  template<typename T> void process(T& obj) {
    if (load) {
      if (pos + sizeof(T) > state.size()) {
        Log(blunted::e_FatalError, "EnvState", "state", "state is invalid");
      }
      memcpy(&obj, &state[pos], sizeof(T));
      pos += sizeof(T);
    } else {
      state.resize(pos + sizeof(T));
      memcpy(&state[pos], &obj, sizeof(T));
      if (!failure && disable_cnt == 0 && !reference.empty() && (*(T*) &state[pos]) != (*(T*) &reference[pos])) {
        failure = true;
        std::cout << "Position:  " << pos << std::endl;
        std::cout << "Type:      " << typeid(obj).name() << std::endl;
        std::cout << "Value:     " << obj << std::endl;
        std::cout << "Reference: " << (*(T*) &reference[pos]) << std::endl;
        if (crash) {
          Log(blunted::e_FatalError, "EnvState", "state", "Reference mismatch");
        } else {
          print_stacktrace();
        }
      }
      pos += sizeof(T);
      if (pos > 10000000) {
        Log(blunted::e_FatalError, "EnvState", "state", "state is too big");
      }
    }
  }
  void SetPlayers(const std::vector<Player*>& players);
  void SetHumanControllers(const std::vector<HumanGamer*>& controllers);
  void SetControllers(const std::vector<AIControlledKeyboard*>& controllers);
  void SetAnimations(const std::vector<Animation*>& animations);
  void SetTeams(Team* team0, Team* team1);
  const std::string& GetState();
 protected:
  bool failure = false;
  bool stack = true;
  bool load = false;
  char disable_cnt = 0;
  bool crash = false;
  std::vector<Player*> players;
  std::vector<Animation*> animations;
  std::vector<Team*> teams;
  std::vector<HumanGamer*> human_controllers;
  std::vector<AIControlledKeyboard*> controllers;
  std::string state;
  std::string reference;
  int pos = 0;
  ScenarioConfig* scenario_config;
  GameContext* context;
 private:
  void process(void** collection, int size, void*& element);
};

struct FormationEntry {
  FormationEntry() { DO_VALIDATION;}
  // Constructor accepts environment coordinates.
  FormationEntry(float x, float y, e_PlayerRole role, bool lazy,
                 bool controllable)
      : position(x, y * FORMATION_Y_SCALE, 0),
        start_position(x, y * FORMATION_Y_SCALE, 0),
        role(role),
        lazy(lazy),
        controllable(controllable) {
    DO_VALIDATION;
  }
  bool operator == (const FormationEntry& f) const {
    return role == f.role &&
        lazy == f.lazy &&
        position == f.position &&
        controllable == f.controllable;
  }
  Vector3 position_env() { DO_VALIDATION;
    return Vector3(position.coords[0],
                   position.coords[1] / FORMATION_Y_SCALE,
                   position.coords[2]);
  }
  void ProcessState(EnvState* state) { DO_VALIDATION;
    state->process(role);
    state->process(position);
    state->process(start_position);
    state->process(lazy);
    state->process(controllable);
  }
  Vector3 position; // adapted to player role (combination of databasePosition and hardcoded role position)
  Vector3 start_position;
  e_PlayerRole role = e_PlayerRole_GK;
  bool lazy = false; // Computer doesn't perform any actions for lazy player.
  // Can be controlled by the player?
  bool controllable = true;
};

struct ScenarioConfig{
public:
    static SHARED_PTR<ScenarioConfig> make() {
        return SHARED_PTR<ScenarioConfig>(new ScenarioConfig());
    }
    bool DynamicPlayerSelection() {
        ComputeCache();
        return cached_dynamic_player_selection;
    }
    int ControllableLeftPlayers() {
        ComputeCache();
        return cached_controllable_left_players;
    }
    int ControllableRightPlayers() {
        ComputeCache();
        return cached_controllable_right_players;
    }
    bool LeftTeamOwnsBall() { DO_VALIDATION;
        float leftDistance = 1000000;
        float rightDistance = 1000000;
        for (auto& player : left_team) { DO_VALIDATION;
            leftDistance = std::min(leftDistance,
                (player.start_position - ball_position).GetLength());
        }
        for (auto& player : right_team) { DO_VALIDATION;
            rightDistance = std::min(rightDistance,
            (player.start_position - ball_position).GetLength());
        }
        return leftDistance < rightDistance;
    }
    void ProcessStateConstant(EnvState* state) {
        cache_computed = false;
        state->process(ball_position);
        int size = left_team.size();
        state->process(size);
        left_team.resize(size);
        size = right_team.size();
        state->process(size);
        right_team.resize(size);
        state->process(left_agents);
        state->process(right_agents);
        state->process(use_magnet);
        state->process(offsides);
        state->process(left_team_difficulty);
        state->process(right_team_difficulty);
        state->process(deterministic);
        state->process(end_episode_on_score);
        state->process(end_episode_on_possession_change);
        state->process(end_episode_on_out_of_play);
        state->process(game_duration);
        state->process(second_half);
        state->process(control_all_players);
    }
    void ProcessState(EnvState* state) {
        cache_computed = false;
        state->process(real_time);
        state->process(game_engine_random_seed);
        state->process(reverse_team_processing);
        for (auto& p : left_team) {
            p.ProcessState(state);
        }
        for (auto& p : right_team) {
            p.ProcessState(state);
        }
    }
    Vector3 ball_position;
    std::vector<FormationEntry> left_team;
    std::vector<FormationEntry> right_team;
    int left_agents = 1;
    int right_agents = 0;
    bool use_magnet = true;
    bool offsides = true;
    bool real_time = false;
    unsigned int game_engine_random_seed = 42;
    bool reverse_team_processing = false;
    float left_team_difficulty = 1.0;
    float right_team_difficulty = 0.6;
    bool deterministic = false;
    bool end_episode_on_score = false;
    bool end_episode_on_possession_change = false;
    bool end_episode_on_out_of_play = false;
    int game_duration = 3000;
    bool control_all_players = false;
    int second_half = 999999999;
    ScenarioConfig() { }
private:
    void ComputeCache() {
        if(cache_computed){
            return;
        }
        cached_controllable_left_players = 0;
        cached_controllable_right_players = 0;
        for(auto& p : left_team){
            if(p.controllable){
                cached_controllable_left_players++;
            }
        }
        for (auto& p : right_team) {
            if (p.controllable) {
            cached_controllable_right_players++;
            }
        }
        cached_dynamic_player_selection =
            !((cached_controllable_left_players == left_agents || left_agents == 0) &&
            (cached_controllable_right_players == right_agents || right_agents == 0));
        cache_computed = true;
    }
    int cached_controllable_left_players = -1;
    int cached_controllable_right_players = -1;
    bool cached_dynamic_player_selection = false;
    bool cache_computed = false;
};

class GameConfig {
 public:
  static SHARED_PTR<GameConfig> make() {
    return SHARED_PTR<GameConfig>(new GameConfig());
  }
  // Is rendering enabled.
  bool render = false;
  // Directory with textures and other resources.
  std::string data_dir;
  // How many physics animation steps are done per single environment step.
  int physics_steps_per_frame = 10;
  int render_resolution_x = 1280;
  int render_resolution_y = 720;
  std::string updatePath(const std::string& path) {
#ifdef WIN32
    boost::filesystem::path boost_path(path);
    if (boost_path.is_absolute()) {
      return path;
    }
    boost::filesystem::path data_dir_boost(data_dir);
    data_dir_boost /= boost_path;
    return data_dir_boost.string();
#else
    if (path[0] == '/') {
      return path;
    }
    return data_dir + '/' + path;
#endif
  }
  void ProcessState(EnvState* state) {
    state->process(data_dir);
    state->process(physics_steps_per_frame);
    state->process(render_resolution_x);
    state->process(render_resolution_y);
  }
 private:
  GameConfig() { }
  friend GameEnv;
};

enum GameState {
  game_created,
  game_initiated,
  game_running,
  game_done
};

Tracker* GetTracker();
GameEnv* GetGame();
GameContext& GetContext();
class Tracker {
 public:
  void setup(long start, long end) {
    this->start = start;
    this->end = end;
    GetContext().tracker_disabled = 0;
    GetContext().tracker_pos = 0;
  }
  void setDisabled(bool disabled) {
    GetContext().tracker_disabled += disabled ? 1 : -1;
  }
  bool enabled() {
    return GetContext().tracker_disabled == 0;
  }
  inline void verify(int line, const char* file) {
    if (GetContext().tracker_disabled) return;
    GetContext().tracker_pos++;
    if (GetContext().tracker_pos < start || GetContext().tracker_pos > end) return;
    std::unique_lock<std::mutex> lock(mtx);
    std::string trace;
    if (waiting_game == nullptr) {
      if (GetContext().tracker_pos % 10000 == 0) {
        std::cout << "Validating: " << GetContext().tracker_pos << std::endl;
      }
      waiting_stack_trace = trace;
      waiting_game = GetGame();
      waiting_line = line;
      waiting_file = file;
      cv.wait(lock);
      return;
    }
    GetContext().tracker_disabled++;
    verify_snapshot(GetContext().tracker_pos, line, file, trace);
    GetContext().tracker_disabled--;
    waiting_game = nullptr;
    cv.notify_one();
  }
 private:
  void verify_snapshot(long pos, int line, const char* file, const std::string& trace);
  // Tweak start and end to verify that line numbers match for
  // each call in the verification range (2 bytes / call).
  long start = 0LL;
  long end = 1000000000LL;
  bool verify_stack_trace = true;
  std::mutex mtx;
  std::condition_variable cv;
  GameEnv* waiting_game = nullptr;
  int waiting_line;
  const char* waiting_file;
  std::string waiting_stack_trace;
};

struct GameEnv {
    GameEnv() {DO_VALIDATION};
    void start_game();
    sharedInfo get_info();
    screenshoot get_frame();

    bool sticky_action_state(int action, bool left_team, int player);
    void action(int action, bool left_team, int player);
    void reset(ScenarioConfig& game_config, bool init_animation);
    void render(bool swap_buffer = true);
    std::string get_state(const std::string& pickle);
    std::string set_state(const std::string& state);
    void tracker_setup(long start, long end) { GetTracker()->setup(start, end); }
    void step();
    void ProcessState(EnvState* state);
    ScenarioConfig& config();
private:
    void setConfig(ScenarioConfig& scenario_config);
    void do_step(int count);
    void getObservations();
    AIControlledKeyboard* keyboard_ = nullptr;
    bool disable_graphics_ = false;
    int last_step_rendered_frames_ = 1;
public:
    ScenarioConfig scenario_config;
    GameConfig game_config;
    GameContext* context = nullptr;
    GameState state = game_created;
    int waiting_for_game_count = 0;
};
