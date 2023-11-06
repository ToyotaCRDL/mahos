var msgpack = require("msgpackr");
var zmq = require("zeromq");
var requester = zmq.socket("req");

var x = 0;
requester.on("message", function(reply) {
    const rep = msgpack.unpack(reply);
    console.log(
      `product: ${rep.product}`,
      `quotient: ${rep.quotient}`,
    );

    x += 1;
    if (x === 3) {
        requester.close();
        process.exit(0);
    }
});

requester.connect("tcp://127.0.0.1:5567");

for (var i = 0; i < 3; i++) {
    var req = msgpack.pack({"a" : 2 * i, "b" : 3});
    requester.send(req);
}

process.on('SIGINT', function() { requester.close(); });
