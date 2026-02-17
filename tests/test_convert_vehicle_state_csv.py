from io import StringIO

from scripts.convert_vehicle_state_csv import convert_rows, map_actor_type


def test_map_actor_type() -> None:
    assert map_actor_type("vehicle.tesla.model3") == "vehicle"
    assert map_actor_type("walker.pedestrian.001") == "walker"
    assert map_actor_type("static.prop") == "static.prop"


def test_convert_rows_reduces_columns() -> None:
    source = StringIO(
        "frame,id,type,location_x,location_y,location_z,ignored\n"
        "100,42,vehicle.tesla.model3,1.0,2.0,3.0,x\n"
    )
    destination = StringIO()

    convert_rows(source, destination)
    converted = destination.getvalue().strip().splitlines()

    assert converted[0] == "frame,id,type,x,y,z"
    assert converted[1] == "100,42,vehicle,1.0,2.0,3.0"
