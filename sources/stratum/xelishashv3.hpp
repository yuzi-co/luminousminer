#pragma once

#include <network/network.hpp>
#include <stratum/stratum.hpp>


namespace stratum
{
    // Official XELIS stratum (vipor-net/xelis-stratum-protocol): JSON-RPC 2.0, LF-delimited.
    //   subscribe  -> result [ session, extranonceHex, len, pubKeyHex ]
    //   notify     params    [ jobId, timestampHex, headerWorkHashHex(32B), algo, clean ]
    //   submit     params    [ worker, jobId, nonceHex(8B BE) ]
    //   set_difficulty params [ diff ]   (target = U256::MAX / diff, big-endian)
    class StratumXelisHashV3 : public stratum::Stratum
    {
      public:
        void onResponse(boost::json::object const& root) final;
        void onMiningNotify(boost::json::object const& root) final;
        void onMiningSetDifficulty(boost::json::object const& root) final;
        void onMiningSetExtraNonce(boost::json::object const& root) final;
        void onUnknownMethod(boost::json::object const& root) final;

        void miningSubscribe() final;
        void miningAuthorize() final;
        void miningSubmit(uint32_t const deviceId, boost::json::array const& params) final;
    };
}
