var msgpack = require("msgpackr");
var zmq = require("zeromq");
var sock = zmq.socket("sub");

sock.connect("tcp://127.0.0.1:5566");
sock.subscribe("datetime");
console.log("Subscribing to datetime of clock");

sock.on("message", function(topic, message) {
    msg = msgpack.unpack(message);
    console.log(`${msg.year}-${msg.month}-${msg.day} ${msg.hour}:${msg.minute}:${msg.second}`);
});
