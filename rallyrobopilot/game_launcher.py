from rallyrobopilot import Car, Track, SunLight, MultiRaySensor
from ursina import *

from rallyrobopilot.checkpoint_manager import CheckpointManager


def prepare_game_app(track_name = "SimpleTrack", init_car = False):
    from ursina import window, Ursina
    
    # Create Window
    window.vsync = False # Set to false to uncap FPS limit of 60
    app = Ursina(size=(600, 400))
    print("Asset folder")
    print(application.asset_folder)

    # Set assets folder. Here assets are one folder up from current location.
    application.asset_folder = application.asset_folder.parent
    print("Asset folder")
    print(application.asset_folder)

    window.title = "Rally"
    window.borderless = False
    window.show_ursina_splash = False
    window.cog_button.disable()
    window.fps_counter.enable()
    window.exit_button.disable()
    
    #   Global models & textures
    #                   car model       particle model    raycast model
    global_models = [ "assets/cars/sports-car.obj", "assets/particles/particles.obj",  "assets/utils/line.obj"]
    #                Car texture             Particle Textures
    global_texs = [ "assets/cars/garage/sports-car/sports-red.png", "sports-blue.png", "sports-green.png", "sports-orange.png", "sports-white.png", "particle_forest_track.png", "red.png"]
    
    # load asset
    track = Track(track_name)
    print("loading assets after track creation")
    track.load_assets(global_models, global_texs)
    
    car = None
    if init_car:
        car = Car()
        car.sports_car()
        car.set_track(track)
        car.multiray_sensor = MultiRaySensor(car, 15, 90)
        car.multiray_sensor.enable()
        car.multiray_sensor.set_enabled_rays(False)
        car.visible = True
        car.enable()
        car.camera_angle = "top"
        car.change_camera = True
        car.camera_follow = True
    
    # Lighting + shadows
    #sun = SunLight(direction = (-0.7, -0.9, 0.5), resolution = 3072, car = car)
    ambient = AmbientLight(color = Vec4(0.5, 0.55, 0.66, 0) * 0.75)
    
    render.setShaderAuto()
    
    # Sky
    Sky(texture = "sky")
    
    mouse.locked = False
    mouse.visible = True
    
    track.activate()
    track.played = True

    checkpoint_manager = CheckpointManager()
    checkpoint_manager.enable()
    checkpoint_manager.add_entities()
   
    return app, car, track
