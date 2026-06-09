#pragma once


#include <network/network.hpp>
#include <stratum/stratum.hpp>


namespace stratum
{
    // Iron Fish stratum (protocol version 3). Envelope: { id, method, body }.
    //   C->S mining.subscribe { version, name, publicAddress }
    //   S->C mining.subscribed { clientId, xn }
    //   S->C mining.set_target { target }            (32-byte big-endian)
    //   S->C mining.notify     { miningRequestId, header }   (180-byte header hex)
    //   S->C mining.wait_for_work
    //   C->S mining.submit     { miningRequestId, randomness } (full 8-byte BE hex)
    //   S->C mining.submitted  { id, result, message? }
    class StratumFishhash : public stratum::Stratum
    {
      public:
        void onResponse(boost::json::object const& root) final;
        void onMiningNotify(boost::json::object const& root) final;
        void onMiningSetDifficulty(boost::json::object const& root) final;
        void onMiningSetTarget(boost::json::object const& root) final;
        void onUnknownMethod(boost::json::object const& root) final;

        void onConnect() final;
        void miningSubscribe() override;
        void miningSubmit(uint32_t const deviceId, boost::json::object const& params) final;
    };
}
